"""Hyperparameter tuning for XGBoost classifier.

Runs a randomized search over key hyperparameters, evaluates on val set,
then reports best test set performance.

Usage:
    python -m depwatch.model_training.tune_xgboost
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from depwatch.common.features import FeatureVector
from depwatch.model_training.evaluate import fast_concordance_index

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters")
    parser.add_argument(
        "--input",
        default="data/training/direct_features.parquet",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    import xgboost as xgb

    df = pd.read_parquet(args.input)
    feature_cols = [c for c in FeatureVector.feature_names() if c in df.columns]
    df["response_time_trend"] = df["response_time_trend"].fillna(1.0)
    df = df.drop_duplicates("repo_name", keep="last").copy()
    df["label"] = df["is_abandoned"].astype(int)

    # Split (same seed as train_baselines)
    rng = np.random.default_rng(42)
    repos = df["repo_name"].unique()
    rng.shuffle(repos)
    n = len(repos)
    train = df[df["repo_name"].isin(set(repos[: int(n * 0.7)]))]
    val = df[df["repo_name"].isin(set(repos[int(n * 0.7) : int(n * 0.85)]))]
    test = df[df["repo_name"].isin(set(repos[int(n * 0.85) :]))]

    x_train = train[feature_cols].values.astype(np.float32)
    y_train = train["label"].values
    x_val = val[feature_cols].values.astype(np.float32)
    y_val = val["label"].values
    x_test = test[feature_cols].values.astype(np.float32)
    y_test = test["label"].values

    # Time-to-event for C-index
    time_test = np.where(
        test["is_abandoned"],
        1.0,
        np.maximum(test["age_months"], 1.0),
    ).astype(np.float32)
    event_test = test["is_abandoned"].astype(int).values

    # Randomized search
    trial_rng = np.random.default_rng(123)
    best_auc = 0.0
    best_params: dict[str, float | int] = {}

    logger.info("Running %d random trials...", args.n_trials)
    for i in range(args.n_trials):
        params = {
            "max_depth": int(trial_rng.choice([4, 6, 8, 10])),
            "learning_rate": float(trial_rng.choice([0.01, 0.03, 0.05, 0.1])),
            "subsample": float(trial_rng.choice([0.7, 0.8, 0.9, 1.0])),
            "colsample_bytree": float(trial_rng.choice([0.6, 0.7, 0.8, 0.9])),
            "min_child_weight": int(trial_rng.choice([1, 3, 5, 10])),
            "gamma": float(trial_rng.choice([0.0, 0.1, 0.5, 1.0])),
            "reg_alpha": float(trial_rng.choice([0.0, 0.01, 0.1, 1.0])),
            "reg_lambda": float(trial_rng.choice([0.5, 1.0, 2.0, 5.0])),
        }

        model = xgb.XGBClassifier(
            n_estimators=1000,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=42,
            tree_method="hist",
            **params,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

        y_prob = model.predict_proba(x_val)[:, 1]
        auc = float(roc_auc_score(y_val, y_prob))

        if auc > best_auc:
            best_auc = auc
            best_params = params
            logger.info(
                "  Trial %d/%d: AUROC=%.4f trees=%d %s",
                i + 1,
                args.n_trials,
                auc,
                model.best_iteration,
                params,
            )

    # Evaluate best params on test set
    logger.info("")
    logger.info("Best val AUROC: %.4f", best_auc)
    logger.info("Best params: %s", best_params)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",
        **best_params,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    y_prob_test = model.predict_proba(x_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, y_prob_test))
    c_idx = fast_concordance_index(time_test, y_prob_test, event_test)

    logger.info("")
    logger.info("=" * 60)
    logger.info("TUNED XGBOOST TEST RESULTS")
    logger.info("=" * 60)
    logger.info("AUROC:   %.4f", test_auc)
    logger.info("C-index: %.4f", c_idx)
    logger.info("Xu et al. reference: C-index = 0.846")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""Train point-in-time baselines: logistic regression + XGBoost.

Per-repo evaluation: each repo gets ONE prediction from its last snapshot,
matching Xu et al.'s methodology. Training uses last snapshot per repo
(110K rows, balanced 50/50), not all snapshots (which inflates dataset
size and introduces within-repo correlation).

Usage:
    python -m depwatch.model_training.train_baselines
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from depwatch.common.features import FeatureVector
from depwatch.model_training.evaluate import EvalResult, evaluate_binary

logger = logging.getLogger(__name__)


def prepare_baseline_data(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Prepare per-repo data for point-in-time baselines.

    Takes the LAST snapshot per repo as the single observation,
    matching Xu et al.'s per-repo evaluation methodology.
    """
    feature_cols = [c for c in FeatureVector.feature_names() if c in df.columns]
    logger.info("Using %d features: %s", len(feature_cols), feature_cols[:5])

    # One row per repo: if dataset has multiple snapshots, take the last one
    if "snapshot_month" in df.columns:
        df = df.sort_values("snapshot_month").drop_duplicates("repo_name", keep="last").copy()
    else:
        df = df.drop_duplicates("repo_name", keep="last").copy()

    # Fill NaN features with sensible defaults
    feature_cols_set = set(feature_cols)
    for col in feature_cols_set:
        if col in df.columns and df[col].isna().any():
            default = 1.0 if col in ("response_time_trend", "activity_deviation") else 0.0
            logger.info("Filling %d NaNs in %s with %.1f", df[col].isna().sum(), col, default)
            df[col] = df[col].fillna(default)

    # Binary label: is_abandoned (already per-repo)
    df["label"] = df["is_abandoned"].astype(int)

    # Time-to-event for C-index and AFT
    # For the direct features dataset, ref_date = last event per repo.
    # For abandoned repos: time_to_event is short (repo is at its end).
    # For active repos: time_to_event uses age as a proxy for observation time.
    # The key insight: C-index ranks repos by risk, so relative ordering matters
    # more than absolute values.
    if "snapshot_month" in df.columns:
        snap_dt = pd.to_datetime(df["snapshot_month"])
        abd_dt = pd.to_datetime(df["estimated_abandonment_date"], errors="coerce")
        max_month = snap_dt.max()
        months_to_abd = (abd_dt.dt.year - snap_dt.dt.year) * 12 + abd_dt.dt.month - snap_dt.dt.month
        months_to_end = (max_month.year - snap_dt.dt.year) * 12 + max_month.month - snap_dt.dt.month
        df["time_to_event"] = np.where(
            df["is_abandoned"],
            np.maximum(months_to_abd.fillna(1), 1),
            np.maximum(months_to_end, 1),
        ).astype(np.float32)
    else:
        # Direct features: use age_months as observation time for active repos,
        # small value for abandoned repos (they're at the event point)
        df["time_to_event"] = np.where(
            df["is_abandoned"],
            1.0,  # abandoned repos: event is imminent at ref_date
            np.maximum(df["age_months"], 1.0),  # active repos: censored at age
        ).astype(np.float32)
    df["event_observed"] = df["is_abandoned"].astype(int)

    logger.info(
        "Per-repo dataset: %d repos, label balance: %.1f%% abandoned",
        len(df),
        df["label"].mean() * 100,
    )

    # Split by repo (70/15/15)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.85)

    train = df.iloc[indices[:n_train]]
    val = df.iloc[indices[n_train:n_val]]
    test = df.iloc[indices[n_val:]]

    logger.info(
        "Split: train=%d repos, val=%d repos, test=%d repos",
        len(train),
        len(val),
        len(test),
    )

    return {
        "x_train": train[feature_cols].values.astype(np.float32),
        "y_train": train["label"].values,
        "x_val": val[feature_cols].values.astype(np.float32),
        "y_val": val["label"].values,
        "x_test": test[feature_cols].values.astype(np.float32),
        "y_test": test["label"].values,
        "time_train": train["time_to_event"].values.astype(np.float32),
        "event_train": train["event_observed"].values,
        "time_val": val["time_to_event"].values.astype(np.float32),
        "event_val": val["event_observed"].values,
        "time_test": test["time_to_event"].values.astype(np.float32),
        "event_test": test["event_observed"].values,
    }


def train_logistic(data: dict[str, np.ndarray]) -> EvalResult:
    """Train logistic regression baseline."""
    logger.info("Training logistic regression...")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(data["x_train"])
    x_val = scaler.transform(data["x_val"])
    x_test = scaler.transform(data["x_test"])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    )
    model.fit(x_train, data["y_train"])

    y_prob_val = model.predict_proba(x_val)[:, 1]
    val_result = evaluate_binary(
        "LogReg (val)",
        data["y_val"],
        y_prob_val,
        data["time_val"],
        data["event_val"],
    )
    logger.info(val_result.summary())

    y_prob_test = model.predict_proba(x_test)[:, 1]
    test_result = evaluate_binary(
        "LogReg (test)",
        data["y_test"],
        y_prob_test,
        data["time_test"],
        data["event_test"],
    )
    logger.info(test_result.summary())

    return test_result


def train_xgboost_classifier(
    data: dict[str, np.ndarray],
    save_dir: str | None = None,
) -> EvalResult:
    """Train XGBoost binary classifier baseline."""
    import xgboost as xgb

    logger.info("Training XGBoost classifier...")

    n_pos = data["y_train"].sum()
    n_neg = len(data["y_train"]) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        data["x_train"],
        data["y_train"],
        eval_set=[(data["x_val"], data["y_val"])],
        verbose=False,
    )
    logger.info("XGBoost stopped at %d trees", model.best_iteration)

    y_prob_val = model.predict_proba(data["x_val"])[:, 1]
    val_result = evaluate_binary(
        "XGBoost (val)",
        data["y_val"],
        y_prob_val,
        data["time_val"],
        data["event_val"],
    )
    logger.info(val_result.summary())

    y_prob_test = model.predict_proba(data["x_test"])[:, 1]
    test_result = evaluate_binary(
        "XGBoost (test)",
        data["y_test"],
        y_prob_test,
        data["time_test"],
        data["event_test"],
    )
    logger.info(test_result.summary())

    # Feature importance
    importances = model.feature_importances_
    feature_names = FeatureVector.feature_names()
    top_features = sorted(
        zip(feature_names, importances, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    logger.info("Top 10 features:")
    for name, imp in top_features:
        logger.info("  %-30s %.4f", name, imp)

    # Save model artifact
    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / "model.json"
        model.save_model(str(model_path))
        logger.info("Model saved to %s", model_path)

    return test_result


def train_xgboost_aft(data: dict[str, np.ndarray]) -> EvalResult:
    """Train XGBoost AFT (Accelerated Failure Time) survival model.

    This is Xu et al.'s best model architecture (C-index 0.846).
    Per-repo training: one observation per repo with time-to-event labels.
    """
    import xgboost as xgb

    logger.info("Training XGBoost AFT...")

    def make_aft_dmatrix(
        features: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> xgb.DMatrix:
        dm = xgb.DMatrix(features)
        lower = time.copy()
        upper = np.where(event == 1, time, np.inf).astype(np.float32)
        dm.set_float_info("label_lower_bound", lower)
        dm.set_float_info("label_upper_bound", upper)
        return dm

    dtrain = make_aft_dmatrix(
        data["x_train"],
        data["time_train"],
        data["event_train"],
    )
    dval = make_aft_dmatrix(
        data["x_val"],
        data["time_val"],
        data["event_val"],
    )
    dtest = make_aft_dmatrix(
        data["x_test"],
        data["time_test"],
        data["event_test"],
    )

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    logger.info("XGBoost AFT stopped at %d trees", model.best_iteration)

    # AFT predicts survival time — invert for risk score (shorter = higher risk)
    pred_time_test = model.predict(dtest)
    risk_scores = -pred_time_test

    # C-index on test set
    from depwatch.model_training.evaluate import fast_concordance_index

    c_idx = fast_concordance_index(
        data["time_test"],
        risk_scores,
        data["event_test"],
    )

    # Normalize risk to [0, 1] for AUROC/AUC-PR
    risk_norm = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)

    result = evaluate_binary(
        "XGBoost AFT (test)",
        data["y_test"],
        risk_norm,
        data["time_test"],
        data["event_test"],
    )
    result.c_index = c_idx
    logger.info(result.summary())

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train point-in-time baselines (per-repo evaluation)",
    )
    parser.add_argument(
        "--input",
        default="data/training/direct_features.parquet",
        help="Path to training dataset (default: direct_features.parquet)",
    )
    parser.add_argument(
        "--save-dir",
        default="artifacts/latest",
        help="Directory to save the production model (default: artifacts/latest)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading dataset from %s", args.input)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.input)
    columns = []
    names = []
    for field in table.schema:
        col = table.column(field.name)
        if pa.types.is_date(field.type):
            col = col.cast(pa.string())
        elif pa.types.is_timestamp(field.type) and field.type.tz is not None:
            col = col.cast(pa.timestamp("us"))
        columns.append(col)
        names.append(field.name)
    df = pa.table(columns, names=names).to_pandas()

    logger.info("Dataset: %d rows, %d repos", len(df), df["repo_name"].nunique())

    # Prepare per-repo data
    data = prepare_baseline_data(df)

    # Train baselines
    results: list[EvalResult] = []

    lr_result = train_logistic(data)
    results.append(lr_result)

    xgb_result = train_xgboost_classifier(data, save_dir=args.save_dir)
    results.append(xgb_result)

    aft_result = train_xgboost_aft(data)
    results.append(aft_result)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BASELINE RESULTS (test set, per-repo evaluation)")
    logger.info("=" * 80)
    for r in results:
        logger.info(r.summary())
    logger.info("=" * 80)
    logger.info("Xu et al. reference: C-index = 0.846")

    best_c = max(r.c_index for r in results)
    if best_c < 0.70:
        logger.warning(
            "Best C-index %.4f is significantly below Xu et al. (0.846). "
            "Consider investigating labeling or feature quality.",
            best_c,
        )
    else:
        logger.info("Best C-index: %.4f", best_c)


if __name__ == "__main__":
    main()

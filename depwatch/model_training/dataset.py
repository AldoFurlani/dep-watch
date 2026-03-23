"""Training data utilities — splits, sliding windows, quality checks.

Training data is loaded from local files (Parquet/CSV) exported from
BigQuery, not from a database. The export/feature-computation pipeline
is in data_pipeline/.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from depwatch.common.features import FeatureVector


def load_snapshot_df(path: str) -> pd.DataFrame:
    """Load a labeled snapshot DataFrame from a Parquet or CSV file.

    Expected columns: repo_id, snapshot_month, is_abandoned,
    abandonment_signal, abandonment_date, plus all 24 feature columns.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_by_repo(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by repo_id to prevent data leakage.

    All snapshots for a given repo go into the same split.
    """
    import numpy as np

    repo_ids = np.array(df["repo_id"].unique())
    rng_gen = np.random.default_rng(seed)
    rng_gen.shuffle(repo_ids)

    n = len(repo_ids)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    train_ids = set(repo_ids[:n_train])
    val_ids = set(repo_ids[n_train:n_val])
    test_ids = set(repo_ids[n_val:])

    train_df = df[df["repo_id"].isin(train_ids)].copy()
    val_df = df[df["repo_id"].isin(val_ids)].copy()
    test_df = df[df["repo_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def compute_class_balance(df: pd.DataFrame) -> dict[str, object]:
    """Compute class balance statistics for quality checks."""
    total = len(df["repo_id"].unique())
    abandoned = df[df["is_abandoned"] == True]["repo_id"].nunique()  # noqa: E712
    active = total - abandoned

    return {
        "total_repos": total,
        "abandoned_repos": abandoned,
        "active_repos": active,
        "abandoned_ratio": abandoned / total if total > 0 else 0.0,
        "signal_distribution": (
            df.drop_duplicates("repo_id")["abandonment_signal"].value_counts().to_dict()
        ),
    }


def compute_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feature summary statistics for data quality checks.

    Returns DataFrame with count, mean, std, min, max, null_count per feature.
    """
    feature_cols = [c for c in FeatureVector.feature_names() if c in df.columns]
    stats = df[feature_cols].describe().T
    stats["null_count"] = df[feature_cols].isnull().sum()
    return stats


def create_sliding_windows(
    df: pd.DataFrame,
    *,
    window_size: int = 6,
    label_horizons: tuple[int, int, int] = (3, 6, 12),
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """Create sliding windows for temporal model training.

    For each repo with enough snapshots, produces windows of
    ``window_size`` consecutive monthly feature vectors and corresponding
    abandonment labels at multiple horizons.

    Returns:
        (windows, labels) where:
        - windows: list of [T, D] feature arrays
        - labels: list of [3] binary labels for each horizon
    """
    feature_cols = [c for c in FeatureVector.feature_names() if c in df.columns]

    windows: list[list[list[float]]] = []
    labels: list[list[float]] = []

    for _repo_id, group in df.groupby("repo_id"):
        group = group.sort_values("snapshot_month")
        if len(group) < window_size:
            continue

        is_abandoned = bool(group.iloc[-1]["is_abandoned"])
        abandonment_date = group.iloc[-1].get("abandonment_date")

        feature_rows = group[feature_cols].values.tolist()

        for start in range(len(group) - window_size + 1):
            window = feature_rows[start : start + window_size]
            window_end_month = group.iloc[start + window_size - 1]["snapshot_month"]

            # Compute labels for each horizon
            horizon_labels: list[float] = []
            for months in label_horizons:
                if is_abandoned and abandonment_date is not None:
                    try:
                        end_dt = datetime.fromisoformat(str(window_end_month))
                        abd_dt = datetime.fromisoformat(str(abandonment_date))
                        months_until = (abd_dt.year - end_dt.year) * 12 + (
                            abd_dt.month - end_dt.month
                        )
                        horizon_labels.append(1.0 if 0 <= months_until <= months else 0.0)
                    except (ValueError, TypeError):
                        horizon_labels.append(0.0)
                else:
                    horizon_labels.append(0.0)

            windows.append(window)
            labels.append(horizon_labels)

    return windows, labels

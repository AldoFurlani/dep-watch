"""Compute 24-dim feature vectors from BigQuery exports.

Reads the exported Parquet files (monthly_stats, issue_durations,
pr_durations, author_commits) and produces a final training dataset
with all 24 features per repo per month.

Usage:
    python -m depwatch.model_training.compute_features [--input-dir data/training]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_90D = 3
WINDOW_365D = 12


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for contribution distribution."""
    if len(values) <= 1:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = sorted_vals.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * sorted_vals).sum()) / (n * total) - (n + 1.0) / n)


def _bus_factor(commit_counts: np.ndarray) -> int:
    """Minimum contributors responsible for 50%+ of commits."""
    if len(commit_counts) == 0:
        return 0
    sorted_desc = np.sort(commit_counts)[::-1]
    total = sorted_desc.sum()
    if total == 0:
        return 0
    cumsum = np.cumsum(sorted_desc)
    result = np.searchsorted(cumsum, total / 2, side="left") + 1
    return int(result)


def _months_since_positive(series: pd.Series[float]) -> pd.Series[int]:
    """For each row, count months since the last positive value (vectorized)."""
    is_positive = series > 0
    # Create groups that reset at each positive value
    groups = is_positive.cumsum()
    # Within each group, cumcount gives distance from start
    result = groups.groupby(groups).cumcount()
    # Where the value itself is positive, distance is 0
    result = result.where(~is_positive, 0)
    return result


def _fill_monthly_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every repo has a row for every month in its active range."""
    logger.info("Filling monthly gaps for %d repos...", df["repo_name"].nunique())

    # Get min/max month per repo
    repo_ranges = df.groupby("repo_name")["snapshot_month"].agg(["min", "max"])

    # Build full index of (repo_name, month) pairs
    all_pairs: list[tuple[object, str]] = []
    for repo_name, (min_m, max_m) in repo_ranges.iterrows():
        months = pd.date_range(min_m, max_m, freq="MS").strftime("%Y-%m-%d").tolist()
        all_pairs.extend((repo_name, m) for m in months)

    full_index = pd.DataFrame(all_pairs, columns=["repo_name", "snapshot_month"])

    # Merge with actual data
    result = full_index.merge(df, on=["repo_name", "snapshot_month"], how="left")
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].fillna(0)
    return result


def compute_rolling_stats(monthly_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling-window features from monthly event counts."""
    df = monthly_stats.copy()

    df = _fill_monthly_gaps(df)
    df = df.sort_values(["repo_name", "snapshot_month"]).reset_index(drop=True)

    logger.info("Computing rolling window features...")
    grouped = df.groupby("repo_name", sort=False)

    # Cumulative counts
    df["stars"] = grouped["stars"].cumsum()
    df["forks"] = grouped["forks"].cumsum()
    df["open_issues"] = (
        grouped["issues_opened"].cumsum() - grouped["issues_closed"].cumsum()
    ).clip(lower=0)

    # Age in months (vectorized)
    first_month = grouped["snapshot_month"].transform("first")
    snap_dt = pd.to_datetime(df["snapshot_month"])
    first_dt = pd.to_datetime(first_month)
    df["age_months"] = (
        (snap_dt.dt.year - first_dt.dt.year) * 12 + snap_dt.dt.month - first_dt.dt.month
    ).astype(float)

    # 90-day (3-month) rolling sums
    df["commit_count_90d"] = (
        grouped["commit_count"].rolling(WINDOW_90D, min_periods=1).sum().reset_index(drop=True)
    )
    df["issues_opened_90d"] = (
        grouped["issues_opened"].rolling(WINDOW_90D, min_periods=1).sum().reset_index(drop=True)
    )
    issues_closed_90d = (
        grouped["issues_closed"].rolling(WINDOW_90D, min_periods=1).sum().reset_index(drop=True)
    )
    df["issue_close_ratio_90d"] = np.minimum(
        np.where(df["issues_opened_90d"] > 0, issues_closed_90d / df["issues_opened_90d"], 0.0),
        1.0,
    )

    df["prs_opened_90d"] = (
        grouped["prs_opened"].rolling(WINDOW_90D, min_periods=1).sum().reset_index(drop=True)
    )
    prs_merged_90d = (
        grouped["prs_merged"].rolling(WINDOW_90D, min_periods=1).sum().reset_index(drop=True)
    )
    df["pr_merge_ratio_90d"] = np.minimum(
        np.where(df["prs_opened_90d"] > 0, prs_merged_90d / df["prs_opened_90d"], 0.0),
        1.0,
    )

    # 365-day (12-month) rolling sums
    commits_365d = (
        grouped["commit_count"].rolling(WINDOW_365D, min_periods=1).sum().reset_index(drop=True)
    )
    df["release_count_365d"] = (
        grouped["releases"].rolling(WINDOW_365D, min_periods=1).sum().reset_index(drop=True)
    )

    # Commit frequency trend: current month vs 12-month average
    # Default to 1.0 (stable) when no baseline, cap extreme values
    monthly_rate_365d = commits_365d / 12.0
    df["commit_frequency_trend"] = np.where(
        monthly_rate_365d > 0, df["commit_count"] / monthly_rate_365d, 1.0
    ).clip(0.0, 100.0)

    # Days since last commit (vectorized)
    df["_months_since_commit"] = grouped["commit_count"].transform(_months_since_positive)
    df["days_since_last_commit"] = df["_months_since_commit"] * 30.44

    # Days since last release (vectorized)
    df["_months_since_release"] = grouped["releases"].transform(_months_since_positive)
    df["days_since_last_release"] = df["_months_since_release"] * 30.44
    df["has_recent_release"] = (df["_months_since_release"] < WINDOW_90D).astype(float)

    # Activity deviation
    activity_current = df["commit_count"] + df["issues_opened"] + df["prs_opened"]
    activity_90d = df["commit_count_90d"] + df["issues_opened_90d"] + df["prs_opened_90d"]
    expected_monthly = activity_90d / 3.0
    df["activity_deviation"] = np.where(
        expected_monthly > 0, activity_current / expected_monthly, 1.0
    ).clip(0.0, 100.0)

    result_cols = [
        "repo_name",
        "snapshot_month",
        "stars",
        "forks",
        "open_issues",
        "age_months",
        "commit_count_90d",
        "days_since_last_commit",
        "commit_frequency_trend",
        "issues_opened_90d",
        "issue_close_ratio_90d",
        "prs_opened_90d",
        "pr_merge_ratio_90d",
        "release_count_365d",
        "days_since_last_release",
        "has_recent_release",
        "activity_deviation",
    ]
    return df[result_cols].copy()


def compute_median_close_times(
    features_df: pd.DataFrame,
    issue_durations: pd.DataFrame,
) -> pd.DataFrame:
    """Add median_issue_close_time_days and response_time_trend features."""
    df = features_df.copy()
    df["median_issue_close_time_days"] = 0.0
    df["response_time_trend"] = 1.0

    if issue_durations.empty:
        return df

    issues = issue_durations.copy()
    # Cap extreme outliers (issues open for years then closed)
    issues["close_time_days"] = issues["close_time_days"].clip(upper=365.0)
    issues["closed_month"] = pd.to_datetime(issues["closed_at"]).dt.to_period("M")

    # Expand each issue to the 3 snapshot months it contributes to (closed month + next 2)
    issues["_month_int"] = issues["closed_month"].apply(lambda p: p.year * 12 + p.month - 1)
    expanded = issues.loc[issues.index.repeat(3)].copy()
    expanded["_offset"] = np.tile([0, 1, 2], len(issues))
    expanded["_snap_int"] = expanded["_month_int"] + expanded["_offset"]
    expanded["_snap_period"] = expanded["_snap_int"].apply(
        lambda x: pd.Period(year=x // 12, month=x % 12 + 1, freq="M")
    )
    expanded["snap_month_str"] = expanded["_snap_period"].astype(str) + "-01"

    # Compute median close time per (repo, snapshot_month)
    median_by_snap = (
        expanded.groupby(["repo_name", "snap_month_str"])["close_time_days"]
        .median()
        .reset_index()
        .rename(columns={"close_time_days": "_median_close"})
    )

    # For response_time_trend: split into recent (offset 0-1) vs older (offset 2)
    recent_median = (
        expanded[expanded["_offset"] <= 1]
        .groupby(["repo_name", "snap_month_str"])["close_time_days"]
        .median()
        .reset_index()
        .rename(columns={"close_time_days": "_recent_median"})
    )
    older_median = (
        expanded[expanded["_offset"] == 2]
        .groupby(["repo_name", "snap_month_str"])["close_time_days"]
        .median()
        .reset_index()
        .rename(columns={"close_time_days": "_older_median"})
    )

    # Merge into features
    df["_snap_key"] = pd.to_datetime(df["snapshot_month"]).dt.strftime("%Y-%m-01")

    df = df.merge(
        median_by_snap,
        left_on=["repo_name", "_snap_key"],
        right_on=["repo_name", "snap_month_str"],
        how="left",
    )
    df["median_issue_close_time_days"] = df["_median_close"].fillna(0.0)

    df = df.merge(
        recent_median,
        left_on=["repo_name", "_snap_key"],
        right_on=["repo_name", "snap_month_str"],
        how="left",
        suffixes=("", "_r"),
    )
    df = df.merge(
        older_median,
        left_on=["repo_name", "_snap_key"],
        right_on=["repo_name", "snap_month_str"],
        how="left",
        suffixes=("", "_o"),
    )
    has_both = (
        df["_recent_median"].notna() & df["_older_median"].notna() & (df["_older_median"] > 0.01)
    )
    df.loc[has_both, "response_time_trend"] = (
        df.loc[has_both, "_recent_median"] / df.loc[has_both, "_older_median"]
    ).clip(0.01, 100.0)  # Cap extreme ratios

    # Clean up temp columns
    drop_cols = [c for c in df.columns if c.startswith("_") or c.startswith("snap_month_str")]
    df = df.drop(columns=drop_cols)

    return df


def compute_median_merge_times(
    features_df: pd.DataFrame,
    pr_durations: pd.DataFrame,
) -> pd.DataFrame:
    """Add median_pr_merge_time_days feature."""
    df = features_df.copy()
    df["median_pr_merge_time_days"] = 0.0

    if pr_durations.empty:
        return df

    prs = pr_durations.copy()
    prs["merge_time_days"] = prs["merge_time_days"].clip(upper=365.0)
    prs["merged_month"] = pd.to_datetime(prs["merged_at"]).dt.to_period("M")

    # Expand each PR to 3 snapshot months
    prs["_month_int"] = prs["merged_month"].apply(lambda p: p.year * 12 + p.month - 1)
    expanded = prs.loc[prs.index.repeat(3)].copy()
    expanded["_offset"] = np.tile([0, 1, 2], len(prs))
    expanded["_snap_int"] = expanded["_month_int"] + expanded["_offset"]
    expanded["_snap_period"] = expanded["_snap_int"].apply(
        lambda x: pd.Period(year=x // 12, month=x % 12 + 1, freq="M")
    )
    expanded["snap_month_str"] = expanded["_snap_period"].astype(str) + "-01"

    median_by_snap = (
        expanded.groupby(["repo_name", "snap_month_str"])["merge_time_days"]
        .median()
        .reset_index()
        .rename(columns={"merge_time_days": "_median_merge"})
    )

    df["_snap_key"] = pd.to_datetime(df["snapshot_month"]).dt.strftime("%Y-%m-01")
    df = df.merge(
        median_by_snap,
        left_on=["repo_name", "_snap_key"],
        right_on=["repo_name", "snap_month_str"],
        how="left",
    )
    df["median_pr_merge_time_days"] = df["_median_merge"].fillna(0.0)

    drop_cols = [c for c in df.columns if c.startswith("_") or c.startswith("snap_month_str")]
    df = df.drop(columns=drop_cols)

    return df


def compute_contributor_features(
    features_df: pd.DataFrame,
    author_commits: pd.DataFrame,
) -> pd.DataFrame:
    """Add contributor health features from per-author commit data.

    Uses incremental computation: iterates snapshots chronologically per repo,
    maintaining a running cumulative state instead of re-filtering for every
    snapshot. ~10x faster than the naive approach.
    """
    import bisect

    df = features_df.copy()
    df["snapshot_month"] = pd.to_datetime(df["snapshot_month"]).dt.date

    df["contributor_count_total"] = 0.0
    df["contributor_count_90d"] = 0.0
    df["top1_contributor_ratio"] = 1.0
    df["bus_factor"] = 0.0
    df["new_contributor_count_90d"] = 0.0
    df["contributor_gini"] = 0.0

    if author_commits.empty:
        return df

    ac = author_commits.copy()
    ac["month"] = pd.to_datetime(ac["month"]).dt.date

    # Pre-build per-repo monthly author dicts
    repo_monthly: dict[str, dict[object, dict[str, int]]] = {}
    repo_first_appearance: dict[str, dict[str, object]] = {}

    for repo_name, group in ac.groupby("repo_name"):
        monthly: dict[object, dict[str, int]] = {}
        first_app: dict[str, object] = {}
        for _, row in group.iterrows():
            m = row["month"]
            author = row["actor_login"]
            count = int(row["commit_count"])
            if m not in monthly:
                monthly[m] = {}
            monthly[m][author] = monthly[m].get(author, 0) + count
            if author not in first_app:
                first_app[author] = m
        repo_monthly[repo_name] = monthly
        repo_first_appearance[repo_name] = first_app

    feature_groups = df.groupby("repo_name")
    total_repos = len(feature_groups)

    for i, (repo_name, repo_features) in enumerate(feature_groups):
        if (i + 1) % 10000 == 0:
            logger.info("  contributor features: %d/%d repos...", i + 1, total_repos)

        if repo_name not in repo_monthly:
            continue

        monthly = repo_monthly[repo_name]
        first_app = repo_first_appearance[repo_name]
        data_months = sorted(monthly.keys(), key=str)

        # Iterate snapshots chronologically with incremental cumulative state
        cumulative: dict[str, int] = {}
        data_ptr = 0

        repo_features = repo_features.sort_values("snapshot_month")
        snap_indices = repo_features.index.values
        snap_months = repo_features["snapshot_month"].values

        for snap_idx, snap in zip(snap_indices, snap_months, strict=True):
            # Update cumulative with any data months <= snap (amortized O(1))
            while data_ptr < len(data_months) and data_months[data_ptr] <= snap:
                for author, count in monthly[data_months[data_ptr]].items():
                    cumulative[author] = cumulative.get(author, 0) + count
                data_ptr += 1

            if not cumulative:
                continue

            counts = np.array(list(cumulative.values()), dtype=np.int64)

            df.at[snap_idx, "contributor_count_total"] = float(len(counts))
            df.at[snap_idx, "contributor_gini"] = _gini_coefficient(counts)
            df.at[snap_idx, "bus_factor"] = float(_bus_factor(counts))

            total = int(counts.sum())
            if total > 0:
                df.at[snap_idx, "top1_contributor_ratio"] = float(counts.max()) / total

            # 90d window: authors active in [window_start, snap]
            snap_date = pd.Timestamp(snap)
            window_start = (snap_date - pd.DateOffset(months=2)).date()

            wi_start = bisect.bisect_left(data_months, window_start)
            wi_end = bisect.bisect_right(data_months, snap)

            recent_authors: set[str] = set()
            for mi in range(wi_start, wi_end):
                recent_authors.update(monthly[data_months[mi]].keys())

            df.at[snap_idx, "contributor_count_90d"] = float(len(recent_authors))

            # New contributors: first appeared in the 90d window
            new_count = sum(1 for a in recent_authors if first_app[a] >= window_start)
            df.at[snap_idx, "new_contributor_count_90d"] = float(new_count)

    return df


def assemble_training_dataset(input_dir: str | Path) -> pd.DataFrame:
    """Full pipeline: read exports -> compute all 24 features -> merge labels."""
    inp = Path(input_dir)

    logger.info("Loading exported data from %s", inp)
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _read_parquet(path: Path) -> pd.DataFrame:
        """Read Parquet, casting date/timestamp columns to avoid db-dtypes issues."""
        table = pq.read_table(path)
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
        table = pa.table(columns, names=names)
        return table.to_pandas()

    monthly_stats = _read_parquet(inp / "monthly_stats.parquet")
    issue_durations = _read_parquet(inp / "issue_durations.parquet")
    pr_durations = _read_parquet(inp / "pr_durations.parquet")
    author_commits = _read_parquet(inp / "author_commits.parquet")
    labels = _read_parquet(inp / "repo_labels.parquet")

    logger.info(
        "Data loaded: %d repos, %d monthly rows",
        monthly_stats["repo_name"].nunique(),
        len(monthly_stats),
    )

    # Step 1: Rolling window features from monthly stats
    logger.info("Computing rolling window features...")
    df = compute_rolling_stats(monthly_stats)

    # Step 2: Median issue close times + response time trend
    logger.info("Computing median issue close times...")
    df = compute_median_close_times(df, issue_durations)

    # Step 3: Median PR merge times
    logger.info("Computing median PR merge times...")
    df = compute_median_merge_times(df, pr_durations)

    # Step 4: Contributor health features
    logger.info("Computing contributor features...")
    df = compute_contributor_features(df, author_commits)

    # Step 5: Merge labels
    logger.info("Merging labels...")
    label_cols = ["repo_name", "is_abandoned", "abandonment_signal", "estimated_abandonment_date"]
    df = df.merge(labels[label_cols], on="repo_name", how="left")
    df["is_abandoned"] = df["is_abandoned"].fillna(False)
    df["abandonment_signal"] = df["abandonment_signal"].fillna("not_abandoned")

    logger.info(
        "Final dataset: %d rows, %d repos, %d features",
        len(df),
        df["repo_name"].nunique(),
        24,
    )

    return df


def main() -> None:
    """CLI entry point for feature computation."""
    parser = argparse.ArgumentParser(description="Compute 24-dim features from BigQuery exports")
    parser.add_argument(
        "--input-dir",
        default="data/training",
        help="Directory with exported Parquet files (default: data/training)",
    )
    parser.add_argument(
        "--output",
        default="data/training/training_dataset.parquet",
        help="Output path for final training dataset",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = assemble_training_dataset(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Training dataset saved -> %s", out_path)


if __name__ == "__main__":
    main()

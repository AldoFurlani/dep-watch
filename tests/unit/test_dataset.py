"""Tests for training data export and dataset utilities."""

import pandas as pd

from depwatch.common.features import FeatureVector
from depwatch.model_training.dataset import (
    compute_class_balance,
    compute_feature_stats,
    create_sliding_windows,
    split_by_repo,
)


def _make_df(n_repos: int = 10, snapshots_per_repo: int = 6) -> pd.DataFrame:
    """Create a synthetic labeled snapshot DataFrame."""
    feature_names = FeatureVector.feature_names()
    records = []
    for i in range(n_repos):
        is_abandoned = i < n_repos // 3  # ~33% abandoned
        for m in range(snapshots_per_repo):
            record = {
                "repo_id": f"repo-{i}",
                "snapshot_month": f"2024-{m + 1:02d}-01",
                "is_abandoned": is_abandoned,
                "abandonment_signal": "inactivity" if is_abandoned else "not_abandoned",
                "abandonment_date": "2024-06-01" if is_abandoned else None,
            }
            for fname in feature_names:
                record[fname] = float(i + m)
            records.append(record)
    return pd.DataFrame(records)


class TestSplitByRepo:
    def test_no_repo_leakage(self) -> None:
        df = _make_df(n_repos=20)
        train, val, test = split_by_repo(df)

        train_ids = set(train["repo_id"].unique())
        val_ids = set(val["repo_id"].unique())
        test_ids = set(test["repo_id"].unique())

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_repos_present(self) -> None:
        df = _make_df(n_repos=20)
        train, val, test = split_by_repo(df)

        all_ids = (
            set(train["repo_id"].unique())
            | set(val["repo_id"].unique())
            | set(test["repo_id"].unique())
        )
        assert all_ids == set(df["repo_id"].unique())

    def test_approximate_split_ratios(self) -> None:
        df = _make_df(n_repos=100)
        train, val, test = split_by_repo(df)

        n_train = train["repo_id"].nunique()
        n_val = val["repo_id"].nunique()
        n_test = test["repo_id"].nunique()

        assert 60 <= n_train <= 80  # ~70
        assert 10 <= n_val <= 20  # ~15
        assert 10 <= n_test <= 20  # ~15

    def test_deterministic_with_seed(self) -> None:
        df = _make_df(n_repos=20)
        t1, v1, _te1 = split_by_repo(df, seed=42)
        t2, v2, _te2 = split_by_repo(df, seed=42)

        assert set(t1["repo_id"].unique()) == set(t2["repo_id"].unique())
        assert set(v1["repo_id"].unique()) == set(v2["repo_id"].unique())


class TestClassBalance:
    def test_balance_stats(self) -> None:
        df = _make_df(n_repos=9, snapshots_per_repo=3)
        stats = compute_class_balance(df)

        assert stats["total_repos"] == 9
        assert stats["abandoned_repos"] == 3
        assert stats["active_repos"] == 6
        assert abs(stats["abandoned_ratio"] - 1 / 3) < 0.01  # type: ignore[operator]

    def test_signal_distribution(self) -> None:
        df = _make_df(n_repos=9)
        stats = compute_class_balance(df)

        dist = stats["signal_distribution"]
        assert isinstance(dist, dict)
        assert "inactivity" in dist or "not_abandoned" in dist


class TestFeatureStats:
    def test_has_expected_columns(self) -> None:
        df = _make_df(n_repos=5)
        stats = compute_feature_stats(df)

        assert "mean" in stats.columns
        assert "null_count" in stats.columns
        assert "stars" in stats.index

    def test_no_nulls_in_synthetic_data(self) -> None:
        df = _make_df(n_repos=5)
        stats = compute_feature_stats(df)
        assert (stats["null_count"] == 0).all()


class TestSlidingWindows:
    def test_window_shape(self) -> None:
        df = _make_df(n_repos=3, snapshots_per_repo=8)
        windows, labels = create_sliding_windows(df, window_size=6)

        assert len(windows) > 0
        assert len(windows) == len(labels)
        assert len(windows[0]) == 6  # T=6
        assert len(windows[0][0]) == FeatureVector.dim()
        assert len(labels[0]) == 3  # 3 horizons

    def test_insufficient_snapshots_skipped(self) -> None:
        df = _make_df(n_repos=3, snapshots_per_repo=4)
        windows, _labels = create_sliding_windows(df, window_size=6)
        assert len(windows) == 0

    def test_labels_are_binary(self) -> None:
        df = _make_df(n_repos=3, snapshots_per_repo=8)
        _, labels = create_sliding_windows(df, window_size=6)

        for label_set in labels:
            for val in label_set:
                assert val in (0.0, 1.0)

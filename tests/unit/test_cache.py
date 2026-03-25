"""Tests for the SQLite snapshot cache."""

from pathlib import Path

import pytest

from depwatch.common.features import FeatureVector
from depwatch.inference_service.services.cache import SnapshotCache


def _make_features(**overrides: float) -> FeatureVector:
    """Create a FeatureVector with sensible defaults."""
    defaults = {name: 0.0 for name in FeatureVector.model_fields}
    defaults.update(overrides)
    return FeatureVector(**defaults)


@pytest.fixture
def cache(tmp_path: Path) -> SnapshotCache:
    db_path = str(tmp_path / "test_cache.db")
    c = SnapshotCache(db_path)
    yield c
    c.close()


class TestSnapshotCache:
    def test_put_and_get(self, cache: SnapshotCache) -> None:
        features = _make_features(stars=1000, forks=200, age_months=36.0)
        cache.put("pallets", "flask", "2025-01", features)

        result = cache.get("pallets", "flask", "2025-01")
        assert result is not None
        assert result.stars == 1000
        assert result.forks == 200
        assert result.age_months == 36.0

    def test_get_missing(self, cache: SnapshotCache) -> None:
        assert cache.get("nonexistent", "repo", "2025-01") is None

    def test_overwrite(self, cache: SnapshotCache) -> None:
        cache.put("o", "r", "2025-01", _make_features(stars=100))
        cache.put("o", "r", "2025-01", _make_features(stars=200))

        result = cache.get("o", "r", "2025-01")
        assert result is not None
        assert result.stars == 200

    def test_case_insensitive_keys(self, cache: SnapshotCache) -> None:
        cache.put("Pallets", "Flask", "2025-01", _make_features(stars=500))
        result = cache.get("pallets", "flask", "2025-01")
        assert result is not None
        assert result.stars == 500

    def test_different_months(self, cache: SnapshotCache) -> None:
        cache.put("o", "r", "2025-01", _make_features(stars=100))
        cache.put("o", "r", "2025-02", _make_features(stars=200))

        r1 = cache.get("o", "r", "2025-01")
        r2 = cache.get("o", "r", "2025-02")
        assert r1 is not None and r1.stars == 100
        assert r2 is not None and r2.stars == 200

    def test_all_features_roundtrip(self, cache: SnapshotCache) -> None:
        features = _make_features(
            stars=50000,
            forks=10000,
            open_issues=500,
            age_months=120.5,
            commit_count_90d=45,
            days_since_last_commit=3.2,
            commit_frequency_trend=1.5,
            contributor_gini=0.65,
            response_time_trend=0.8,
            activity_deviation=1.2,
            bus_factor=3.0,
            has_recent_release=1.0,
        )
        cache.put("owner", "repo", "2025-03", features)
        result = cache.get("owner", "repo", "2025-03")

        assert result is not None
        assert result.to_list() == features.to_list()

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "persist.db")

        cache1 = SnapshotCache(db_path)
        cache1.put("o", "r", "2025-01", _make_features(stars=999))
        cache1.close()

        cache2 = SnapshotCache(db_path)
        result = cache2.get("o", "r", "2025-01")
        cache2.close()

        assert result is not None
        assert result.stars == 999

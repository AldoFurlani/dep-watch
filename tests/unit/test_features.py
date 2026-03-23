"""Tests for the 24-dimensional FeatureVector model."""

from depwatch.common.features import FeatureVector


class TestFeatureVector:
    def test_dimensionality_is_24(self) -> None:
        assert FeatureVector.dim() == 24

    def test_feature_names_count(self) -> None:
        assert len(FeatureVector.feature_names()) == 24

    def test_to_list_length(self) -> None:
        fv = FeatureVector(**{name: 0.0 for name in FeatureVector.feature_names()})
        assert len(fv.to_list()) == 24

    def test_to_list_preserves_values(self) -> None:
        values = {name: float(i) for i, name in enumerate(FeatureVector.feature_names())}
        fv = FeatureVector(**values)
        result = fv.to_list()
        for i, name in enumerate(FeatureVector.feature_names()):
            assert result[i] == float(i), f"Mismatch at {name}"

    def test_feature_names_order_matches_fields(self) -> None:
        names = FeatureVector.feature_names()
        assert names[0] == "stars"
        assert names[3] == "age_months"
        assert names[-1] == "has_recent_release"

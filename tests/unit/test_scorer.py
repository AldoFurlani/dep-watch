"""Tests for the scorer service."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from depwatch.common.features import FeatureVector
from depwatch.inference_service.services.scorer import (
    Scorer,
    _risk_level,
)


def _make_features(**overrides: float) -> FeatureVector:
    defaults = {name: 0.0 for name in FeatureVector.model_fields}
    defaults.update(overrides)
    return FeatureVector(**defaults)


class TestRiskLevel:
    def test_low(self) -> None:
        assert _risk_level(0.10) == "low"

    def test_medium(self) -> None:
        assert _risk_level(0.30) == "medium"

    def test_high(self) -> None:
        assert _risk_level(0.60) == "high"

    def test_critical(self) -> None:
        assert _risk_level(0.80) == "critical"

    def test_boundaries(self) -> None:
        assert _risk_level(0.0) == "low"
        assert _risk_level(0.25) == "medium"
        assert _risk_level(0.50) == "high"
        assert _risk_level(0.75) == "critical"


class TestScorer:
    def test_load_missing_model(self, tmp_path: Path) -> None:
        scorer = Scorer(str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="Model not found"):
            scorer.load()

    def test_score_without_load(self) -> None:
        scorer = Scorer("dummy")
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer.score("pkg", "pypi", "o", "r", _make_features())

    def test_is_loaded(self) -> None:
        scorer = Scorer("dummy")
        assert scorer.is_loaded is False

    def test_score_with_mock_model(self) -> None:
        """Test scoring with directly injected mock model and explainer."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        # Mock SHAP explainer
        mock_explainer = MagicMock()
        rng = np.random.default_rng(42)
        shap_vals = rng.standard_normal((1, 24))
        mock_explainer.shap_values.return_value = [
            np.zeros((1, 24)),  # class 0
            shap_vals,  # class 1
        ]

        scorer = Scorer("dummy")
        scorer._model = mock_model
        scorer._explainer = mock_explainer
        assert scorer.is_loaded

        features = _make_features(stars=1000, forks=200, days_since_last_commit=30)
        result = scorer.score("flask", "pypi", "pallets", "flask", features)

        assert result.package_name == "flask"
        assert result.ecosystem == "pypi"
        assert result.abandonment_probability == 0.7
        assert result.risk_level == "high"
        assert len(result.top_risk_factors) == 5
        # Risk factors should be sorted by absolute SHAP value
        abs_shaps = [abs(rf.shap_value) for rf in result.top_risk_factors]
        assert abs_shaps == sorted(abs_shaps, reverse=True)

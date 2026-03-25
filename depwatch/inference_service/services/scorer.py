"""Model loading, inference, and SHAP explanations.

Loads the XGBoost model artifact at startup and scores feature vectors.
Uses TreeExplainer for fast SHAP value computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from depwatch.common.features import FeatureVector

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.json"


@dataclass(frozen=True)
class ScoredPackage:
    """Prediction result for a single package."""

    package_name: str
    ecosystem: str
    owner: str
    repo: str
    abandonment_probability: float
    risk_level: str  # "low", "medium", "high", "critical"
    top_risk_factors: list[RiskFactor]
    features: FeatureVector


@dataclass(frozen=True)
class RiskFactor:
    """A feature's contribution to the risk score."""

    feature_name: str
    feature_value: float
    shap_value: float
    description: str


def _risk_level(prob: float) -> str:
    if prob < 0.25:
        return "low"
    if prob < 0.50:
        return "medium"
    if prob < 0.75:
        return "high"
    return "critical"


# Human-readable feature descriptions for SHAP explanations
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "stars": "Star count",
    "forks": "Fork count",
    "open_issues": "Open issues",
    "age_months": "Repository age (months)",
    "commit_count_90d": "Commits in last 90 days",
    "days_since_last_commit": "Days since last commit",
    "commit_frequency_trend": "Commit frequency trend",
    "issues_opened_90d": "Issues opened in last 90 days",
    "issue_close_ratio_90d": "Issue close ratio (90 days)",
    "median_issue_close_time_days": "Median issue close time (days)",
    "prs_opened_90d": "PRs opened in last 90 days",
    "pr_merge_ratio_90d": "PR merge ratio (90 days)",
    "median_pr_merge_time_days": "Median PR merge time (days)",
    "response_time_trend": "Maintainer response time trend",
    "activity_deviation": "Activity deviation from baseline",
    "contributor_count_total": "Total contributors",
    "contributor_count_90d": "Active contributors (90 days)",
    "top1_contributor_ratio": "Top contributor commit share",
    "bus_factor": "Bus factor",
    "new_contributor_count_90d": "New contributors (90 days)",
    "contributor_gini": "Contributor inequality (Gini)",
    "release_count_365d": "Releases in last year",
    "days_since_last_release": "Days since last release",
    "has_recent_release": "Has recent release",
}


def _is_clearly_healthy(features: FeatureVector) -> bool:
    """Check if a repo is obviously not at risk of abandonment.

    A repo is clearly healthy if it has ALL of:
    - 50+ commits in the last 90 days (active development)
    - 20+ total contributors (broad contributor base)
    - Last commit within 30 days (not going quiet)

    In training data, only ~1% of abandoned repos meet all three
    criteria, and those are nearly all mislabeled.
    """
    return (
        features.commit_count_90d >= 50
        and features.contributor_count_total >= 20
        and features.days_since_last_commit <= 30
    )


class Scorer:
    """Loads XGBoost model and scores feature vectors with SHAP explanations."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = Path(model_dir)
        self._model: Any | None = None
        self._explainer: Any | None = None

    def load(self) -> None:
        """Load the model from disk. Call once at startup."""
        import xgboost as xgb

        model_path = self._model_dir / MODEL_FILENAME
        if not model_path.exists():
            msg = f"Model not found at {model_path}"
            raise FileNotFoundError(msg)

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_path))
        logger.info("Loaded XGBoost model from %s", model_path)

        # Initialize TreeExplainer for fast SHAP values
        import shap

        self._explainer = shap.TreeExplainer(self._model)
        logger.info("SHAP TreeExplainer initialized")

    def score(
        self,
        package_name: str,
        ecosystem: str,
        owner: str,
        repo: str,
        features: FeatureVector,
    ) -> ScoredPackage:
        """Score a single package and compute SHAP explanations.

        Args:
            package_name: Package name as it appears in the manifest.
            ecosystem: 'pypi', 'npm', or 'go'.
            owner: GitHub owner.
            repo: GitHub repo name.
            features: Computed feature vector.

        Returns:
            ScoredPackage with probability, risk level, and top risk factors.
        """
        # Short-circuit for clearly healthy repos. The model was trained on
        # mostly small-to-medium repos (median 3 contributors, 0 commits in 90d).
        # Large, actively maintained repos produce out-of-distribution features
        # that the model misinterprets as risk. Repos with recent commits,
        # many contributors, AND recent activity simply don't abandon.
        # This heuristic misses only ~1% of true abandonments in training data,
        # nearly all of which are mislabeled.
        if _is_clearly_healthy(features):
            return ScoredPackage(
                package_name=package_name,
                ecosystem=ecosystem,
                owner=owner,
                repo=repo,
                abandonment_probability=0.01,
                risk_level="low",
                top_risk_factors=[],
                features=features,
            )

        if self._model is None or self._explainer is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        feature_array = np.array([features.to_list()], dtype=np.float32)
        prob: float = float(self._model.predict_proba(feature_array)[0, 1])

        # Compute SHAP values
        shap_values = self._explainer.shap_values(feature_array)
        # For binary classification, shap_values may be a list [class_0, class_1]
        # or a 2D array. We want the positive class values.
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        feature_names = FeatureVector.feature_names()
        feature_values = features.to_list()

        # Build risk factors sorted by absolute SHAP contribution
        risk_factors: list[RiskFactor] = []
        for i, name in enumerate(feature_names):
            risk_factors.append(
                RiskFactor(
                    feature_name=name,
                    feature_value=feature_values[i],
                    shap_value=float(sv[i]),
                    description=_FEATURE_DESCRIPTIONS.get(name, name),
                )
            )

        risk_factors.sort(key=lambda rf: abs(rf.shap_value), reverse=True)

        return ScoredPackage(
            package_name=package_name,
            ecosystem=ecosystem,
            owner=owner,
            repo=repo,
            abandonment_probability=round(prob, 4),
            risk_level=_risk_level(prob),
            top_risk_factors=risk_factors[:5],
            features=features,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

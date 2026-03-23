"""Sentiment analysis using VADER — no external API calls."""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def compound_score(text: str) -> float:
    """Return VADER compound sentiment score for text (-1.0 to 1.0)."""
    if not text or not text.strip():
        return 0.0
    return float(_get_analyzer().polarity_scores(text)["compound"])


def mean_compound(texts: list[str]) -> float:
    """Return mean VADER compound score across multiple texts."""
    if not texts:
        return 0.0
    scores = [compound_score(t) for t in texts]
    return sum(scores) / len(scores)

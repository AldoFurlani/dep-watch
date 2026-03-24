"""Evaluation metrics for abandonment prediction models.

Primary metric: C-index (concordance index) for comparison with Xu et al.'s 0.846.
Secondary: AUROC, AUC-PR, Brier score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass
class EvalResult:
    """Evaluation metrics for a single model."""

    model_name: str
    auroc: float
    auc_pr: float
    brier: float
    c_index: float

    def summary(self) -> str:
        return (
            f"{self.model_name:25s} | "
            f"AUROC={self.auroc:.4f} | "
            f"AUC-PR={self.auc_pr:.4f} | "
            f"Brier={self.brier:.4f} | "
            f"C-index={self.c_index:.4f}"
        )


def concordance_index(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Compute Harrell's concordance index (C-index).

    Higher predicted_scores should correspond to higher risk (shorter survival).

    Args:
        event_times: Time-to-event (or censoring) for each sample.
        predicted_scores: Model's risk scores (higher = more likely to abandon).
        event_observed: 1 if event occurred, 0 if censored.

    Returns:
        C-index in [0, 1]. 0.5 = random, 1.0 = perfect ranking.
    """
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where the shorter-time subject had an event
            if event_times[i] < event_times[j] and event_observed[i]:
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1
                elif predicted_scores[i] < predicted_scores[j]:
                    discordant += 1
                else:
                    tied += 1
            elif event_times[j] < event_times[i] and event_observed[j]:
                if predicted_scores[j] > predicted_scores[i]:
                    concordant += 1
                elif predicted_scores[j] < predicted_scores[i]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


def fast_concordance_index(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Approximate C-index using sampling for large datasets.

    For N > 50K, the O(N^2) exact computation is too slow.
    Samples 500K random pairs for a reliable estimate.
    """
    n = len(event_times)
    if n <= 5000:
        return concordance_index(event_times, predicted_scores, event_observed)

    rng = np.random.default_rng(42)
    n_pairs = min(500_000, n * (n - 1) // 2)

    concordant = 0
    discordant = 0
    tied = 0
    valid = 0

    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue

        if event_times[i] < event_times[j] and event_observed[i]:
            valid += 1
            if predicted_scores[i] > predicted_scores[j]:
                concordant += 1
            elif predicted_scores[i] < predicted_scores[j]:
                discordant += 1
            else:
                tied += 1
        elif event_times[j] < event_times[i] and event_observed[j]:
            valid += 1
            if predicted_scores[j] > predicted_scores[i]:
                concordant += 1
            elif predicted_scores[j] < predicted_scores[i]:
                discordant += 1
            else:
                tied += 1

    if valid == 0:
        return 0.5
    return (concordant + 0.5 * tied) / valid


def evaluate_binary(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    event_times: np.ndarray | None = None,
    event_observed: np.ndarray | None = None,
) -> EvalResult:
    """Compute all evaluation metrics for a binary classifier.

    Args:
        model_name: Name for display.
        y_true: Binary ground truth (0/1).
        y_prob: Predicted probability of positive class.
        event_times: Optional, for C-index computation.
        event_observed: Optional, for C-index computation.
    """
    auroc = float(roc_auc_score(y_true, y_prob))
    auc_pr = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    c_idx = 0.0
    if event_times is not None and event_observed is not None:
        c_idx = fast_concordance_index(event_times, y_prob, event_observed)

    return EvalResult(
        model_name=model_name,
        auroc=auroc,
        auc_pr=auc_pr,
        brier=brier,
        c_index=c_idx,
    )

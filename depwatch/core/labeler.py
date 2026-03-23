"""Rule-based ground truth labeler for repository abandonment.

Assigns abandonment labels based on configurable thresholds over
observable signals: commit recency, issue activity, archive status,
README keywords, and release recency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from depwatch.common.types import (
        GitHubCommit,
        GitHubIssue,
        GitHubRelease,
        GitHubRepo,
    )


class AbandonmentSignal(StrEnum):
    """The primary signal that triggered the abandonment label."""

    ARCHIVED = "archived"
    README_KEYWORD = "readme_keyword"
    INACTIVITY = "inactivity"
    NOT_ABANDONED = "not_abandoned"


@dataclass
class LabelThresholds:
    """Configurable thresholds for abandonment classification."""

    # Days without commits to consider inactive
    inactivity_days: int = 365
    # Days without issue closure to consider unresponsive
    issue_silence_days: int = 180
    # Days without a release to consider stale
    release_silence_days: int = 365
    # Minimum signals that must agree for inactivity-based label
    min_inactivity_signals: int = 2


# Patterns that indicate abandonment — match "no longer maintained" but NOT
# "we maintain high standards" or "maintained by the community".
_ABANDONMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"no\s+longer\s+maintained", re.IGNORECASE),
    re.compile(r"this\s+project\s+is\s+(now\s+)?archived", re.IGNORECASE),
    re.compile(r"this\s+project\s+is\s+(now\s+)?deprecated", re.IGNORECASE),
    re.compile(r"no\s+longer\s+(actively\s+)?developed", re.IGNORECASE),
    re.compile(r"unmaintained", re.IGNORECASE),
    re.compile(r"this\s+project\s+has\s+been\s+abandoned", re.IGNORECASE),
    re.compile(r"not\s+(being\s+)?actively\s+maintained", re.IGNORECASE),
    re.compile(r"this\s+repo(sitory)?\s+is\s+(now\s+)?unmaintained", re.IGNORECASE),
    re.compile(r"consider\s+this\s+project\s+dead", re.IGNORECASE),
    re.compile(r"project\s+is\s+(now\s+)?in\s+maintenance\s+mode", re.IGNORECASE),
]


@dataclass
class LabelResult:
    """Output of the labeling process."""

    is_abandoned: bool
    signal: AbandonmentSignal
    abandonment_date: datetime | None = None
    details: dict[str, object] = field(default_factory=dict)


def _check_readme_keywords(readme: str | None) -> bool:
    """Return True if README contains abandonment language."""
    if not readme:
        return False
    return any(p.search(readme) for p in _ABANDONMENT_PATTERNS)


def label_repo(
    *,
    repo: GitHubRepo,
    commits: list[GitHubCommit],
    issues: list[GitHubIssue],
    releases: list[GitHubRelease],
    readme_content: str | None,
    thresholds: LabelThresholds | None = None,
    reference_date: datetime | None = None,
) -> LabelResult:
    """Apply rule-based labeling to determine if a repo is abandoned.

    Signal priority:
    1. Archived flag — strongest signal, immediate label.
    2. README keywords — explicit maintainer declaration.
    3. Inactivity heuristics — multiple weak signals must agree.
    """
    th = thresholds or LabelThresholds()
    now = reference_date or datetime.now(tz=UTC)

    # --- Signal 1: Archived ---
    if repo.archived:
        return LabelResult(
            is_abandoned=True,
            signal=AbandonmentSignal.ARCHIVED,
            abandonment_date=repo.updated_at,
            details={"reason": "Repository is archived on GitHub"},
        )

    # --- Signal 2: README keywords ---
    if _check_readme_keywords(readme_content):
        return LabelResult(
            is_abandoned=True,
            signal=AbandonmentSignal.README_KEYWORD,
            abandonment_date=repo.pushed_at or repo.updated_at,
            details={"reason": "README contains abandonment language"},
        )

    # --- Signal 3: Inactivity heuristics ---
    inactivity_signals: list[str] = []

    # 3a: No recent commits
    commit_dates: list[datetime] = [c.committed_at for c in commits if c.committed_at is not None]
    if commit_dates:
        last_commit = max(commit_dates)
        days_since_commit = (now - last_commit).days
        if days_since_commit > th.inactivity_days:
            inactivity_signals.append(f"no_commits_{days_since_commit}d")
    elif repo.pushed_at:
        days_since_push = (now - repo.pushed_at).days
        if days_since_push > th.inactivity_days:
            inactivity_signals.append(f"no_commits_{days_since_push}d")

    # 3b: No recent issue closures
    closed_dates: list[datetime] = [i.closed_at for i in issues if i.closed_at is not None]
    if closed_dates:
        last_closed = max(closed_dates)
        days_since_closed = (now - last_closed).days
        if days_since_closed > th.issue_silence_days:
            inactivity_signals.append(f"no_issue_closures_{days_since_closed}d")
    else:
        # No closed issues at all — only count if there ARE open issues
        open_issues = [i for i in issues if i.state == "open"]
        if open_issues:
            inactivity_signals.append("no_issue_closures_ever")

    # 3c: No recent releases
    release_dates: list[datetime] = [
        r.published_at for r in releases if r.published_at is not None and not r.draft
    ]
    if release_dates:
        last_release = max(release_dates)
        days_since_release = (now - last_release).days
        if days_since_release > th.release_silence_days:
            inactivity_signals.append(f"no_releases_{days_since_release}d")

    if len(inactivity_signals) >= th.min_inactivity_signals:
        # Estimate abandonment date as the most recent activity
        activity_dates = commit_dates + closed_dates + release_dates
        est_abandonment = max(activity_dates) if activity_dates else repo.created_at
        return LabelResult(
            is_abandoned=True,
            signal=AbandonmentSignal.INACTIVITY,
            abandonment_date=est_abandonment,
            details={
                "signals": inactivity_signals,
                "signal_count": len(inactivity_signals),
            },
        )

    return LabelResult(
        is_abandoned=False,
        signal=AbandonmentSignal.NOT_ABANDONED,
        details={"inactivity_signals": inactivity_signals},
    )

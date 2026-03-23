"""Extract 24-dimensional feature vectors from pre-fetched raw data.

The extractor accepts already-fetched GitHub data so it can be
unit-tested trivially without mocking any HTTP calls.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from depwatch.common.features import FeatureVector

if TYPE_CHECKING:
    from depwatch.common.types import (
        GitHubCommit,
        GitHubContributor,
        GitHubIssue,
        GitHubPullRequest,
        GitHubRelease,
        GitHubRepo,
    )


def _days_between(a: datetime, b: datetime) -> float:
    """Absolute days between two datetimes."""
    return abs((a - b).total_seconds()) / 86400.0


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den != 0 else default


def _gini_coefficient(values: list[int]) -> float:
    """Compute Gini coefficient for a list of contribution counts.

    Returns 0.0 for perfectly equal, approaches 1.0 for maximally unequal.
    """
    if len(values) <= 1:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2.0 * cumsum) / (n * total) - (n + 1.0) / n


# Feature caps matching BigQuery training data bounds.
# Without these, truncated API data produces out-of-distribution values.
_MAX_COMMIT_FREQUENCY_TREND = 12.0
_MAX_ACTIVITY_DEVIATION = 3.0
_MAX_RESPONSE_TIME_TREND = 100.0


@dataclass(frozen=True)
class CountOverrides:
    """Accurate event counts that override len()-based counting.

    When the API returns paginated data (e.g., max 100 items), these
    provide the true counts from totalCount fields so features match
    the training distribution.
    """

    commits_30d: int | None = None
    commits_90d: int | None = None
    commits_365d: int | None = None
    issues_90d: int | None = None


def extract_features(
    *,
    repo: GitHubRepo,
    commits: list[GitHubCommit],
    issues: list[GitHubIssue],
    pulls: list[GitHubPullRequest],
    contributors: list[GitHubContributor],
    releases: list[GitHubRelease],
    snapshot_date: datetime | None = None,
    count_overrides: CountOverrides | None = None,
) -> FeatureVector:
    """Build a FeatureVector from pre-fetched raw data.

    Args:
        snapshot_date: Reference point for "N days ago" calculations.
                       Defaults to utcnow.
        count_overrides: Accurate counts from API totalCount fields.
                         When provided, these replace len()-based counting
                         for commit features, preventing train/serve skew
                         from API pagination limits.
    """
    now = snapshot_date or datetime.now(tz=UTC)

    # --- Repo metadata ---
    stars = float(repo.stargazers_count)
    forks = float(repo.forks_count)
    open_issues = float(repo.open_issues_count)
    age_months = max(_days_between(now, repo.created_at) / 30.44, 0.0)

    # --- Commit activity ---
    commit_dates: list[datetime] = [c.committed_at for c in commits if c.committed_at is not None]
    commits_30d = [d for d in commit_dates if (now - d).days <= 30]
    commits_90d = [d for d in commit_dates if (now - d).days <= 90]
    commits_365d = [d for d in commit_dates if (now - d).days <= 365]

    # Use accurate totalCount when available (overrides truncated list counts)
    co = count_overrides
    n_commits_30d = float(co.commits_30d if co and co.commits_30d is not None else len(commits_30d))
    n_commits_90d = float(co.commits_90d if co and co.commits_90d is not None else len(commits_90d))
    n_commits_365d = float(
        co.commits_365d if co and co.commits_365d is not None else len(commits_365d)
    )

    commit_count_90d = n_commits_90d

    if commit_dates:
        most_recent_commit = max(commit_dates)
        days_since_last_commit = max(_days_between(now, most_recent_commit), 0.0)
    else:
        days_since_last_commit = age_months * 30.44

    # Trend: recent 30d rate vs annualized rate
    monthly_rate_recent = n_commits_30d
    monthly_rate_older = _safe_div(n_commits_365d, 12.0)
    commit_frequency_trend = min(
        _safe_div(monthly_rate_recent, monthly_rate_older),
        _MAX_COMMIT_FREQUENCY_TREND,
    )

    # --- Issue activity ---
    issues_opened_90d_list = [i for i in issues if (now - i.created_at).days <= 90]
    issues_closed_90d_list = [
        i for i in issues if i.closed_at is not None and (now - i.closed_at).days <= 90
    ]

    # Use accurate totalCount when available
    issues_opened_90d = float(
        co.issues_90d if co and co.issues_90d is not None else len(issues_opened_90d_list)
    )
    issues_closed_90d_count = len(issues_closed_90d_list)
    issue_close_ratio_90d = min(_safe_div(float(issues_closed_90d_count), issues_opened_90d), 1.0)

    close_times_90d = [
        _days_between(closed, i.created_at)
        for i in issues_closed_90d_list
        if (closed := i.closed_at) is not None
    ]
    median_issue_close_time_days = statistics.median(close_times_90d) if close_times_90d else 0.0

    # --- PR activity ---
    prs_opened_90d_list = [p for p in pulls if (now - p.created_at).days <= 90]
    prs_merged_90d_list = [
        p for p in pulls if p.merged_at is not None and (now - p.merged_at).days <= 90
    ]

    prs_opened_90d = float(len(prs_opened_90d_list))
    prs_merged_90d_count = len(prs_merged_90d_list)
    pr_merge_ratio_90d = min(_safe_div(float(prs_merged_90d_count), prs_opened_90d), 1.0)

    merge_times_90d = [
        _days_between(merged, p.created_at)
        for p in prs_merged_90d_list
        if (merged := p.merged_at) is not None
    ]
    median_pr_merge_time_days = statistics.median(merge_times_90d) if merge_times_90d else 0.0

    # --- Maintainer responsiveness (Xu et al.) ---
    # Response time trend: compare median close time in recent half (0-45d)
    # vs older half (45-90d) of the 90d window.
    recent_close_times = [
        _days_between(closed, i.created_at)
        for i in issues
        if (closed := i.closed_at) is not None and (now - closed).days <= 45
    ]
    older_close_times = [
        _days_between(closed, i.created_at)
        for i in issues
        if (closed := i.closed_at) is not None and 45 < (now - closed).days <= 90
    ]
    median_recent = statistics.median(recent_close_times) if recent_close_times else 0.0
    median_older = statistics.median(older_close_times) if older_close_times else 0.0
    response_time_trend = min(
        _safe_div(median_recent, median_older, default=1.0),
        _MAX_RESPONSE_TIME_TREND,
    )

    # Activity deviation: ratio of actual 30d activity to expected from 90d
    # Use accurate commit counts when available
    activity_30d = (
        n_commits_30d
        + float(len([i for i in issues if (now - i.created_at).days <= 30]))
        + float(len([p for p in pulls if (now - p.created_at).days <= 30]))
    )
    activity_90d = (
        n_commits_90d + float(len(issues_opened_90d_list)) + float(len(prs_opened_90d_list))
    )
    expected_monthly = _safe_div(activity_90d, 3.0)
    activity_deviation = min(
        _safe_div(activity_30d, expected_monthly, default=1.0),
        _MAX_ACTIVITY_DEVIATION,
    )

    # --- Contributor health ---
    contributor_count_total = float(len(contributors))

    commit_authors_90d = {
        c.author_login
        for c in commits
        if c.author_login and c.committed_at is not None and (now - c.committed_at).days <= 90
    }
    contributor_count_90d = float(len(commit_authors_90d))

    if contributors:
        total_contributions = sum(c.contributions for c in contributors)
        top1 = max(c.contributions for c in contributors)
        top1_contributor_ratio = _safe_div(float(top1), float(total_contributions))

        sorted_contribs = sorted([c.contributions for c in contributors], reverse=True)
        cumsum = 0.0
        bus_factor = 0.0
        half = total_contributions / 2.0
        for count in sorted_contribs:
            cumsum += count
            bus_factor += 1.0
            if cumsum >= half:
                break
    else:
        top1_contributor_ratio = 1.0
        bus_factor = 0.0

    older_authors = {
        c.author_login
        for c in commits
        if c.author_login and c.committed_at is not None and (now - c.committed_at).days > 90
    }
    new_authors_90d = commit_authors_90d - older_authors if commit_authors_90d else set()
    new_contributor_count_90d = float(len(new_authors_90d))

    contributor_gini = _gini_coefficient([c.contributions for c in contributors])

    # --- Release metrics ---
    non_draft_releases = [r for r in releases if not r.draft]
    release_dates = [r.published_at for r in non_draft_releases if r.published_at is not None]

    releases_365d = [d for d in release_dates if (now - d).days <= 365]
    release_count_365d = float(len(releases_365d))

    if release_dates:
        most_recent_release = max(release_dates)
        days_since_last_release = max(_days_between(now, most_recent_release), 0.0)
    else:
        days_since_last_release = age_months * 30.44

    releases_90d = [d for d in release_dates if (now - d).days <= 90]
    has_recent_release = 1.0 if releases_90d else 0.0

    return FeatureVector(
        stars=stars,
        forks=forks,
        open_issues=open_issues,
        age_months=age_months,
        commit_count_90d=commit_count_90d,
        days_since_last_commit=days_since_last_commit,
        commit_frequency_trend=commit_frequency_trend,
        issues_opened_90d=issues_opened_90d,
        issue_close_ratio_90d=issue_close_ratio_90d,
        median_issue_close_time_days=median_issue_close_time_days,
        prs_opened_90d=prs_opened_90d,
        pr_merge_ratio_90d=pr_merge_ratio_90d,
        median_pr_merge_time_days=median_pr_merge_time_days,
        response_time_trend=response_time_trend,
        activity_deviation=activity_deviation,
        contributor_count_total=contributor_count_total,
        contributor_count_90d=contributor_count_90d,
        top1_contributor_ratio=top1_contributor_ratio,
        bus_factor=bus_factor,
        new_contributor_count_90d=new_contributor_count_90d,
        contributor_gini=contributor_gini,
        release_count_365d=release_count_365d,
        days_since_last_release=days_since_last_release,
        has_recent_release=has_recent_release,
    )

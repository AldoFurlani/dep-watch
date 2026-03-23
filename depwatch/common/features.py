"""24-dimensional feature vector for repository abandonment prediction.

Feature design informed by Xu et al. (2025) "Predicting Abandonment of
Open Source Software Projects with An Integrated Feature Framework"
(arxiv 2507.21678), which found maintainer-behavior and contributor-diversity
features to be the strongest predictors (C-index 0.748 → 0.846).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    """Monthly snapshot of 24 behavioral features for a repository.

    Features are grouped into categories:
    - Repo metadata (1-4)
    - Commit activity (5-7)
    - Issue activity (8-10)
    - PR activity (11-13)
    - Maintainer responsiveness (14-15)
    - Contributor health (16-21)
    - Release metrics (22-24)
    """

    # --- Repo metadata ---
    stars: float = Field(description="Stargazers count")
    forks: float = Field(description="Fork count")
    open_issues: float = Field(description="Open issue count from GitHub")
    age_months: float = Field(description="Months since repo creation")

    # --- Commit activity ---
    commit_count_90d: float = Field(description="Commits in last 90 days")
    days_since_last_commit: float = Field(description="Days since most recent commit")
    commit_frequency_trend: float = Field(
        description="Ratio of recent (30d) to older (365d/12) commit rate. "
        ">1 = accelerating. Kept for logistic regression baseline."
    )

    # --- Issue activity ---
    issues_opened_90d: float = Field(description="Issues opened in last 90 days")
    issue_close_ratio_90d: float = Field(
        description="Closed / opened issues in last 90 days. Capped at 1.0."
    )
    median_issue_close_time_days: float = Field(
        description="Median days to close issues (closed in last 90 days)"
    )

    # --- PR activity ---
    prs_opened_90d: float = Field(description="PRs opened in last 90 days")
    pr_merge_ratio_90d: float = Field(
        description="Merged / opened PRs in last 90 days. Capped at 1.0."
    )
    median_pr_merge_time_days: float = Field(
        description="Median days to merge PRs (merged in last 90 days)"
    )

    # --- Maintainer responsiveness (Xu et al.) ---
    response_time_trend: float = Field(
        description="Ratio of recent (0-45d) to older (45-90d) median issue "
        "close times. >1 = maintainers slowing down."
    )
    activity_deviation: float = Field(
        description="Ratio of recent 30d activity (commits+issues+PRs) to "
        "expected monthly rate from 90d baseline. <1 = declining."
    )

    # --- Contributor health ---
    contributor_count_total: float = Field(description="Total unique contributors")
    contributor_count_90d: float = Field(description="Unique commit authors in last 90 days")
    top1_contributor_ratio: float = Field(description="Fraction of commits by top contributor")
    bus_factor: float = Field(description="Minimum contributors responsible for 50%+ of commits")
    new_contributor_count_90d: float = Field(
        description="Contributors whose first commit is within last 90 days"
    )
    contributor_gini: float = Field(
        description="Gini coefficient of contribution distribution. "
        "0 = perfectly equal, 1 = one person does everything."
    )

    # --- Release metrics ---
    release_count_365d: float = Field(description="Releases in last 365 days")
    days_since_last_release: float = Field(description="Days since most recent release")
    has_recent_release: float = Field(description="1.0 if a release in last 90 days, else 0.0")

    def to_list(self) -> list[float]:
        """Return feature values as a flat list in field-definition order."""
        return [getattr(self, name) for name in type(self).model_fields]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return ordered list of feature names."""
        return list(cls.model_fields.keys())

    @classmethod
    def dim(cls) -> int:
        """Return feature dimensionality."""
        return len(cls.model_fields)

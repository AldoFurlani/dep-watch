"""Tests for the feature extractor — raw data → 24-dim FeatureVector."""

from datetime import UTC, datetime, timedelta

from depwatch.common.features import FeatureVector
from depwatch.common.types import (
    GitHubCommit,
    GitHubContributor,
    GitHubIssue,
    GitHubPullRequest,
    GitHubRelease,
    GitHubRepo,
)
from depwatch.core.feature_extractor import (
    _gini_coefficient,
    extract_features,
)

NOW = datetime(2025, 6, 1, tzinfo=UTC)


def _make_repo(**overrides: object) -> GitHubRepo:
    defaults = {
        "id": 1,
        "name": "test-repo",
        "full_name": "owner/test-repo",
        "owner_login": "owner",
        "description": "A test repo",
        "language": "Python",
        "stargazers_count": 1000,
        "forks_count": 200,
        "open_issues_count": 50,
        "watchers_count": 800,
        "archived": False,
        "fork": False,
        "created_at": datetime(2020, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2025, 5, 20, tzinfo=UTC),
        "pushed_at": datetime(2025, 5, 25, tzinfo=UTC),
        "default_branch": "main",
        "license_name": "MIT",
        "has_wiki": True,
        "has_issues": True,
    }
    defaults.update(overrides)
    return GitHubRepo(**defaults)  # type: ignore[arg-type]


def _make_commit(days_ago: int, author: str = "dev1") -> GitHubCommit:
    return GitHubCommit(
        sha=f"abc{days_ago}",
        author_login=author,
        committer_login=author,
        message=f"commit {days_ago} days ago",
        committed_at=NOW - timedelta(days=days_ago),
    )


def _make_issue(
    days_ago_created: int,
    days_ago_closed: int | None = None,
    comments: int = 2,
) -> GitHubIssue:
    created = NOW - timedelta(days=days_ago_created)
    closed = NOW - timedelta(days=days_ago_closed) if days_ago_closed is not None else None
    return GitHubIssue(
        number=days_ago_created,
        title=f"Issue from {days_ago_created}d ago",
        state="closed" if closed else "open",
        user_login="user1",
        created_at=created,
        updated_at=created,
        closed_at=closed,
        comments=comments,
        is_pull_request=False,
    )


def _make_pr(
    days_ago_created: int,
    days_ago_merged: int | None = None,
) -> GitHubPullRequest:
    created = NOW - timedelta(days=days_ago_created)
    merged = NOW - timedelta(days=days_ago_merged) if days_ago_merged is not None else None
    return GitHubPullRequest(
        number=days_ago_created,
        title=f"PR from {days_ago_created}d ago",
        state="closed" if merged else "open",
        user_login="user1",
        created_at=created,
        updated_at=created,
        closed_at=merged,
        merged_at=merged,
    )


def _make_release(days_ago: int, draft: bool = False) -> GitHubRelease:
    return GitHubRelease(
        tag_name=f"v{days_ago}",
        name=f"Release {days_ago}d ago",
        published_at=NOW - timedelta(days=days_ago),
        draft=draft,
        prerelease=False,
    )


def _extract(**overrides: object) -> FeatureVector:
    """Helper: extract features with sensible defaults, overridable."""
    kwargs: dict[str, object] = {
        "repo": _make_repo(),
        "commits": [],
        "issues": [],
        "pulls": [],
        "contributors": [],
        "releases": [],
        "snapshot_date": NOW,
    }
    kwargs.update(overrides)
    return extract_features(**kwargs)  # type: ignore[arg-type]


class TestBasicShape:
    def test_returns_24_features(self) -> None:
        fv = _extract(
            commits=[_make_commit(5)],
            contributors=[GitHubContributor(login="dev1", contributions=100)],
        )
        assert isinstance(fv, FeatureVector)
        assert len(fv.to_list()) == 24

    def test_empty_repo_no_errors(self) -> None:
        """Brand new repo with zero activity should not error."""
        fv = _extract(
            repo=_make_repo(
                stargazers_count=0,
                forks_count=0,
                open_issues_count=0,
                watchers_count=0,
            ),
        )
        assert all(v >= 0.0 for v in fv.to_list())


class TestRepoMetadata:
    def test_star_and_fork_counts(self) -> None:
        fv = _extract(repo=_make_repo(stargazers_count=500, forks_count=100))
        assert fv.stars == 500.0
        assert fv.forks == 100.0


class TestCommitActivity:
    def test_commit_count_90d(self) -> None:
        commits = [
            _make_commit(5),  # within 90d
            _make_commit(25),  # within 90d
            _make_commit(60),  # within 90d
            _make_commit(200),  # outside 90d
        ]
        fv = _extract(
            commits=commits,
            contributors=[GitHubContributor(login="dev1", contributions=100)],
        )
        assert fv.commit_count_90d == 3.0

    def test_days_since_last_commit(self) -> None:
        fv = _extract(
            commits=[_make_commit(10)],
            contributors=[GitHubContributor(login="dev1", contributions=1)],
        )
        assert abs(fv.days_since_last_commit - 10.0) < 0.1

    def test_commit_frequency_trend(self) -> None:
        # 2 commits in 30d, 4 total in 365d → trend = 2 / (4/12) = 6.0
        commits = [
            _make_commit(5),
            _make_commit(20),
            _make_commit(100),
            _make_commit(300),
        ]
        fv = _extract(
            commits=commits,
            contributors=[GitHubContributor(login="dev1", contributions=4)],
        )
        assert fv.commit_frequency_trend > 1.0  # accelerating


class TestIssueActivity:
    def test_issue_close_ratio_capped_at_1(self) -> None:
        issues = [
            _make_issue(10, days_ago_closed=5),
            _make_issue(200, days_ago_closed=20),
        ]
        fv = _extract(issues=issues)
        assert fv.issue_close_ratio_90d <= 1.0

    def test_median_issue_close_time(self) -> None:
        # Issue closed in 2 days and issue closed in 10 days → median = 6.0
        issues = [
            _make_issue(30, days_ago_closed=28),  # 2 days to close
            _make_issue(40, days_ago_closed=30),  # 10 days to close
        ]
        fv = _extract(issues=issues)
        assert abs(fv.median_issue_close_time_days - 6.0) < 0.1


class TestPRActivity:
    def test_pr_merge_ratio(self) -> None:
        pulls = [
            _make_pr(10, days_ago_merged=5),
            _make_pr(20),  # not merged
        ]
        fv = _extract(pulls=pulls)
        assert fv.pr_merge_ratio_90d == 0.5

    def test_median_pr_merge_time(self) -> None:
        # PR created 20d ago, merged 15d ago → 5 days
        # PR created 30d ago, merged 20d ago → 10 days
        # median = 7.5
        pulls = [
            _make_pr(20, days_ago_merged=15),
            _make_pr(30, days_ago_merged=20),
        ]
        fv = _extract(pulls=pulls)
        assert abs(fv.median_pr_merge_time_days - 7.5) < 0.1


class TestMaintainerResponsiveness:
    def test_response_time_trend_slowing(self) -> None:
        # Recent issues (0-45d) close slowly, older (45-90d) close fast
        issues = [
            _make_issue(50, days_ago_closed=48),  # closed in 2d, in older window
            _make_issue(20, days_ago_closed=10),  # closed in 10d, in recent window
        ]
        fv = _extract(issues=issues)
        assert fv.response_time_trend > 1.0  # slowing down

    def test_response_time_trend_no_data(self) -> None:
        fv = _extract(issues=[])
        assert fv.response_time_trend == 1.0  # default

    def test_activity_deviation_declining(self) -> None:
        # 90d has 9 commits, but 30d has only 1 → expected 3/month, actual 1
        commits = [
            _make_commit(5),  # 30d window
            *[_make_commit(d) for d in range(40, 90, 6)],  # 8 more in 40-90d
        ]
        fv = _extract(
            commits=commits,
            contributors=[GitHubContributor(login="dev1", contributions=9)],
        )
        assert fv.activity_deviation < 1.0  # declining

    def test_activity_deviation_stable(self) -> None:
        # Even distribution across 90d
        commits = [_make_commit(d) for d in [10, 20, 50, 60, 70, 80]]
        fv = _extract(
            commits=commits,
            contributors=[GitHubContributor(login="dev1", contributions=6)],
        )
        assert 0.5 < fv.activity_deviation < 2.0  # roughly stable


class TestContributorHealth:
    def test_contributor_metrics(self) -> None:
        contributors = [
            GitHubContributor(login="dev1", contributions=80),
            GitHubContributor(login="dev2", contributions=15),
            GitHubContributor(login="dev3", contributions=5),
        ]
        fv = _extract(
            commits=[_make_commit(5, "dev1"), _make_commit(10, "dev2")],
            contributors=contributors,
        )
        assert fv.contributor_count_total == 3.0
        assert fv.top1_contributor_ratio == 0.8
        assert fv.bus_factor == 1.0  # dev1 alone has > 50%

    def test_bus_factor_multiple(self) -> None:
        contributors = [
            GitHubContributor(login="a", contributions=30),
            GitHubContributor(login="b", contributions=30),
            GitHubContributor(login="c", contributions=30),
            GitHubContributor(login="d", contributions=10),
        ]
        fv = _extract(contributors=contributors)
        assert fv.bus_factor == 2.0  # a + b = 60 > 50

    def test_contributor_gini_equal(self) -> None:
        contributors = [
            GitHubContributor(login="a", contributions=10),
            GitHubContributor(login="b", contributions=10),
            GitHubContributor(login="c", contributions=10),
        ]
        fv = _extract(contributors=contributors)
        assert fv.contributor_gini < 0.05  # near 0 = equal

    def test_contributor_gini_concentrated(self) -> None:
        contributors = [
            GitHubContributor(login="a", contributions=100),
            GitHubContributor(login="b", contributions=1),
            GitHubContributor(login="c", contributions=1),
        ]
        fv = _extract(contributors=contributors)
        assert fv.contributor_gini > 0.5  # highly concentrated

    def test_contributor_gini_empty(self) -> None:
        fv = _extract(contributors=[])
        assert fv.contributor_gini == 0.0


class TestGiniCoefficient:
    def test_perfect_equality(self) -> None:
        assert _gini_coefficient([10, 10, 10, 10]) == 0.0

    def test_maximum_inequality(self) -> None:
        gini = _gini_coefficient([0, 0, 0, 100])
        assert gini > 0.7

    def test_single_contributor(self) -> None:
        assert _gini_coefficient([50]) == 0.0

    def test_empty(self) -> None:
        assert _gini_coefficient([]) == 0.0

    def test_two_unequal(self) -> None:
        gini = _gini_coefficient([1, 99])
        assert 0.4 < gini < 0.6


class TestReleaseMetrics:
    def test_release_metrics(self) -> None:
        releases = [
            _make_release(10),
            _make_release(200),
            _make_release(400, draft=True),  # draft, excluded
        ]
        fv = _extract(releases=releases)
        assert fv.has_recent_release == 1.0
        assert abs(fv.days_since_last_release - 10.0) < 0.1

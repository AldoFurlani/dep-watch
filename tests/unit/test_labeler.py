"""Tests for the rule-based abandonment labeler."""

from datetime import UTC, datetime, timedelta

from depwatch.common.types import (
    GitHubCommit,
    GitHubIssue,
    GitHubRelease,
    GitHubRepo,
)
from depwatch.core.labeler import (
    AbandonmentSignal,
    LabelThresholds,
    label_repo,
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
        "stargazers_count": 100,
        "forks_count": 20,
        "open_issues_count": 5,
        "watchers_count": 80,
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


def _commit(days_ago: int) -> GitHubCommit:
    return GitHubCommit(
        sha=f"abc{days_ago}",
        author_login="dev",
        committer_login="dev",
        message="fix",
        committed_at=NOW - timedelta(days=days_ago),
    )


def _issue(days_ago_created: int, days_ago_closed: int | None = None) -> GitHubIssue:
    created = NOW - timedelta(days=days_ago_created)
    closed = NOW - timedelta(days=days_ago_closed) if days_ago_closed is not None else None
    return GitHubIssue(
        number=1,
        title="bug",
        state="closed" if closed else "open",
        user_login="user",
        created_at=created,
        updated_at=created,
        closed_at=closed,
        comments=0,
        is_pull_request=False,
    )


def _release(days_ago: int) -> GitHubRelease:
    return GitHubRelease(
        tag_name="v1",
        name="Release",
        published_at=NOW - timedelta(days=days_ago),
        draft=False,
        prerelease=False,
    )


class TestArchivedSignal:
    def test_archived_repo_is_abandoned(self) -> None:
        result = label_repo(
            repo=_make_repo(archived=True),
            commits=[_commit(5)],
            issues=[],
            releases=[],
            readme_content=None,
            reference_date=NOW,
        )
        assert result.is_abandoned is True
        assert result.signal == AbandonmentSignal.ARCHIVED

    def test_archived_takes_priority_over_activity(self) -> None:
        result = label_repo(
            repo=_make_repo(archived=True),
            commits=[_commit(1)],  # very recent
            issues=[_issue(1, 1)],
            releases=[_release(1)],
            readme_content=None,
            reference_date=NOW,
        )
        assert result.signal == AbandonmentSignal.ARCHIVED


class TestReadmeKeywordSignal:
    def test_no_longer_maintained(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(5)],
            issues=[],
            releases=[],
            readme_content="⚠️ This project is no longer maintained.",
            reference_date=NOW,
        )
        assert result.is_abandoned is True
        assert result.signal == AbandonmentSignal.README_KEYWORD

    def test_unmaintained(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[],
            issues=[],
            releases=[],
            readme_content="UNMAINTAINED: use alternative-package instead",
            reference_date=NOW,
        )
        assert result.is_abandoned is True
        assert result.signal == AbandonmentSignal.README_KEYWORD

    def test_not_actively_maintained(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[],
            issues=[],
            releases=[],
            readme_content="This project is not being actively maintained.",
            reference_date=NOW,
        )
        assert result.is_abandoned is True

    def test_does_not_match_maintain_high_standards(self) -> None:
        """'we maintain high standards' should NOT trigger."""
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(5)],
            issues=[],
            releases=[_release(10)],
            readme_content="We maintain high standards of code quality.",
            reference_date=NOW,
        )
        assert result.is_abandoned is False

    def test_does_not_match_well_maintained(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(5)],
            issues=[],
            releases=[_release(10)],
            readme_content="This is a well-maintained project.",
            reference_date=NOW,
        )
        assert result.is_abandoned is False

    def test_no_readme(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(5)],
            issues=[],
            releases=[],
            readme_content=None,
            reference_date=NOW,
        )
        # No readme keyword signal, may or may not be abandoned by inactivity
        assert result.signal != AbandonmentSignal.README_KEYWORD


class TestInactivitySignal:
    def test_multiple_inactivity_signals_trigger_abandoned(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(400)],  # > 365 days
            issues=[_issue(500, days_ago_closed=400)],  # closed > 180 days ago
            releases=[_release(400)],  # > 365 days
            readme_content=None,
            reference_date=NOW,
        )
        assert result.is_abandoned is True
        assert result.signal == AbandonmentSignal.INACTIVITY

    def test_single_inactivity_signal_not_enough(self) -> None:
        """Default requires 2 signals."""
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(400)],  # stale commits
            issues=[_issue(10, days_ago_closed=5)],  # but recent issue closures
            releases=[_release(10)],  # and recent releases
            readme_content=None,
            reference_date=NOW,
        )
        assert result.is_abandoned is False

    def test_custom_thresholds(self) -> None:
        thresholds = LabelThresholds(
            inactivity_days=30,
            issue_silence_days=30,
            release_silence_days=30,
            min_inactivity_signals=2,
        )
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(60)],
            issues=[_issue(100, days_ago_closed=60)],
            releases=[_release(60)],
            readme_content=None,
            thresholds=thresholds,
            reference_date=NOW,
        )
        assert result.is_abandoned is True

    def test_active_repo_not_abandoned(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(5), _commit(15)],
            issues=[_issue(10, days_ago_closed=3)],
            releases=[_release(30)],
            readme_content="Active project",
            reference_date=NOW,
        )
        assert result.is_abandoned is False
        assert result.signal == AbandonmentSignal.NOT_ABANDONED

    def test_open_issues_with_no_closures_counts(self) -> None:
        """Open issues but never closed anything — one inactivity signal."""
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(400)],  # stale
            issues=[_issue(10)],  # open, never closed
            releases=[],
            readme_content=None,
            reference_date=NOW,
        )
        assert result.is_abandoned is True
        assert "signals" in result.details


class TestLabelResult:
    def test_abandonment_date_set_for_inactivity(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(400)],
            issues=[_issue(500, days_ago_closed=300)],
            releases=[],
            readme_content=None,
            reference_date=NOW,
        )
        assert result.is_abandoned is True
        assert result.abandonment_date is not None

    def test_not_abandoned_has_no_date(self) -> None:
        result = label_repo(
            repo=_make_repo(),
            commits=[_commit(5)],
            issues=[],
            releases=[_release(10)],
            readme_content=None,
            reference_date=NOW,
        )
        assert result.abandonment_date is None

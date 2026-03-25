"""GitHub GraphQL API client for inference.

Fetches all repo data in 1-3 queries instead of 42+ REST calls per package.
Uses the separate GraphQL rate budget (5,000 points/hour).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from depwatch.common.config import Settings

from depwatch.common.types import (
    GitHubCommit,
    GitHubContributor,
    GitHubIssue,
    GitHubPullRequest,
    GitHubRelease,
    GitHubRepo,
)

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

# Single query that fetches all data needed for feature extraction.
# Uses totalCount for accurate commit counts across time windows,
# plus node data for distribution-based features (medians, Gini, etc).
REPO_QUERY = """
query RepoData(
  $owner: String!, $name: String!,
  $since365: GitTimestamp!, $since90: GitTimestamp!, $since30: GitTimestamp!,
  $since90dt: DateTime!
) {
  repository(owner: $owner, name: $name) {
    databaseId
    name
    nameWithOwner
    description
    primaryLanguage { name }
    stargazerCount
    forkCount
    isArchived
    isFork
    createdAt
    updatedAt
    pushedAt
    defaultBranchRef { name }
    licenseInfo { spdxId }
    hasWikiEnabled
    hasIssuesEnabled
    issues(states: OPEN) { totalCount }

    # Accurate commit counts via totalCount (no pagination limit)
    commitCount365: defaultBranchRef {
      target { ... on Commit { history(since: $since365) { totalCount } } }
    }
    commitCount90: defaultBranchRef {
      target { ... on Commit { history(since: $since90) { totalCount } } }
    }
    commitCount30: defaultBranchRef {
      target { ... on Commit { history(since: $since30) { totalCount } } }
    }

    # Recent commit nodes for author extraction and date-based features
    recentCommits: defaultBranchRef {
      target {
        ... on Commit {
          history(first: 100) {
            nodes {
              oid
              message
              committedDate
              author { user { login } }
              committer { user { login } }
            }
          }
        }
      }
    }

    # Older commits for contributor diversity
    olderCommits: defaultBranchRef {
      target {
        ... on Commit {
          history(first: 100, since: $since365) {
            nodes {
              oid
              committedDate
              author { user { login } }
              committer { user { login } }
            }
          }
        }
      }
    }

    # Accurate issue/PR counts via filterBy (no pagination limit)
    issueCount90: issues(filterBy: {since: $since90dt}) { totalCount }
    prCount90: pullRequests(states: [OPEN, CLOSED, MERGED]) { totalCount }

    recentIssues: issues(first: 100, orderBy: {field: UPDATED_AT, direction: DESC},
                         filterBy: {since: $since90dt}) {
      nodes {
        number
        title
        state
        createdAt
        updatedAt
        closedAt
        comments { totalCount }
        author { login }
      }
    }

    recentPRs: pullRequests(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        title
        state
        createdAt
        updatedAt
        closedAt
        mergedAt
        author { login }
      }
    }

    releases(first: 50, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        tagName
        name
        publishedAt
        isDraft
        isPrerelease
      }
    }
  }
}
"""


class GraphQLError(Exception):
    """Raised when the GraphQL API returns errors."""


@dataclass(frozen=True)
class ActivityCounts:
    """Accurate event counts from GraphQL totalCount fields.

    These bypass the 100-node pagination limit to give true counts
    matching the BigQuery training data.
    """

    commits_30d: int
    commits_90d: int
    commits_365d: int
    issues_90d: int | None = None


class GitHubGraphQLClient:
    """Async GitHub GraphQL client for inference.

    Fetches all repo data in a single GraphQL query, dramatically reducing
    API calls compared to REST (1-3 calls vs 42+).
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self._http = http_client
        self._settings = settings

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.github_token:
            headers["Authorization"] = f"Bearer {self._settings.github_token}"
        return headers

    async def _graphql(
        self,
        query: str,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a GraphQL query."""
        response = await self._http.post(
            GRAPHQL_URL,
            headers=self._headers(),
            json={"query": query, "variables": variables},
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()

        if "errors" in result:
            errors = result["errors"]
            msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise GraphQLError(msg)

        data: dict[str, Any] = result["data"]
        return data

    async def fetch_repo_data(
        self,
        owner: str,
        name: str,
    ) -> tuple[
        GitHubRepo,
        list[GitHubCommit],
        list[GitHubIssue],
        list[GitHubPullRequest],
        list[GitHubContributor],
        list[GitHubRelease],
        ActivityCounts,
    ]:
        """Fetch all data needed for feature extraction in a single query.

        Returns (repo, commits, issues, pulls, contributors, releases, counts).
        ActivityCounts provides accurate commit counts from totalCount fields.
        Contributors are approximated from commit authors.
        """
        now = datetime.now(tz=UTC)
        since_365 = (now - timedelta(days=365)).isoformat()
        since_90 = (now - timedelta(days=90)).isoformat()
        since_30 = (now - timedelta(days=30)).isoformat()

        data = await self._graphql(
            REPO_QUERY,
            {
                "owner": owner,
                "name": name,
                "since365": since_365,
                "since90": since_90,
                "since30": since_30,
                "since90dt": since_90,
            },
        )

        repo_data = data["repository"]
        if repo_data is None:
            msg = f"Repository {owner}/{name} not found"
            raise GraphQLError(msg)

        repo = self._parse_repo(repo_data)
        commits = self._parse_commits(repo_data)
        issues = self._parse_issues(repo_data)
        pulls = self._parse_pulls(repo_data)
        contributors = self._approximate_contributors(commits)
        releases = self._parse_releases(repo_data)
        counts = self._parse_counts(repo_data)

        logger.info(
            "Fetched %s/%s: %d commits (actual: 30d=%d, 90d=%d, 365d=%d), "
            "%d issues, %d PRs, %d contributors, %d releases",
            owner,
            name,
            len(commits),
            counts.commits_30d,
            counts.commits_90d,
            counts.commits_365d,
            len(issues),
            len(pulls),
            len(contributors),
            len(releases),
        )

        return repo, commits, issues, pulls, contributors, releases, counts

    @staticmethod
    def _parse_counts(data: dict[str, Any]) -> ActivityCounts:
        """Extract accurate totalCount values for commit/issue/PR time windows."""

        def _get_commit_count(key: str) -> int:
            branch = data.get(key)
            if not branch or not branch.get("target"):
                return 0
            history = branch["target"].get("history", {})
            return int(history.get("totalCount", 0))

        def _get_total_count(key: str) -> int | None:
            section = data.get(key)
            if not section:
                return None
            return int(section.get("totalCount", 0))

        return ActivityCounts(
            commits_30d=_get_commit_count("commitCount30"),
            commits_90d=_get_commit_count("commitCount90"),
            commits_365d=_get_commit_count("commitCount365"),
            issues_90d=_get_total_count("issueCount90"),
        )

    @staticmethod
    def _parse_repo(data: dict[str, Any]) -> GitHubRepo:
        default_branch = "main"
        if data.get("defaultBranchRef"):
            default_branch = data["defaultBranchRef"].get("name", "main")

        return GitHubRepo(
            id=data["databaseId"],
            name=data["name"],
            full_name=data["nameWithOwner"],
            owner_login=data["nameWithOwner"].split("/")[0],
            description=data.get("description"),
            language=(data.get("primaryLanguage") or {}).get("name"),
            stargazers_count=data.get("stargazerCount", 0),
            forks_count=data.get("forkCount", 0),
            open_issues_count=data.get("issues", {}).get("totalCount", 0),
            archived=data.get("isArchived", False),
            fork=data.get("isFork", False),
            created_at=data["createdAt"],
            updated_at=data["updatedAt"],
            pushed_at=data.get("pushedAt"),
            default_branch=default_branch,
            license_name=(data.get("licenseInfo") or {}).get("spdxId"),
            has_wiki=data.get("hasWikiEnabled", False),
            has_issues=data.get("hasIssuesEnabled", True),
        )

    @staticmethod
    def _parse_commits(data: dict[str, Any]) -> list[GitHubCommit]:
        """Parse commits from both recent and older commit history."""
        commits: list[GitHubCommit] = []
        seen_shas: set[str] = set()

        for branch_key in ("recentCommits", "olderCommits"):
            branch = data.get(branch_key)
            if not branch or not branch.get("target"):
                continue
            history = branch["target"].get("history", {})
            for node in history.get("nodes", []):
                sha = node["oid"]
                if sha in seen_shas:
                    continue
                seen_shas.add(sha)

                author_user = (node.get("author") or {}).get("user")
                committer_user = (node.get("committer") or {}).get("user")
                commits.append(
                    GitHubCommit(
                        sha=sha,
                        author_login=author_user["login"] if author_user else None,
                        committer_login=committer_user["login"] if committer_user else None,
                        message=node.get("message", ""),
                        committed_at=node.get("committedDate"),
                    )
                )

        return commits

    @staticmethod
    def _parse_issues(data: dict[str, Any]) -> list[GitHubIssue]:
        issues: list[GitHubIssue] = []
        for node in (data.get("recentIssues") or {}).get("nodes", []):
            issues.append(
                GitHubIssue(
                    number=node["number"],
                    title=node["title"],
                    state=node["state"].lower(),
                    user_login=(node.get("author") or {}).get("login"),
                    created_at=node["createdAt"],
                    updated_at=node["updatedAt"],
                    closed_at=node.get("closedAt"),
                    comments=(node.get("comments") or {}).get("totalCount", 0),
                    is_pull_request=False,
                )
            )
        return issues

    @staticmethod
    def _parse_pulls(data: dict[str, Any]) -> list[GitHubPullRequest]:
        pulls: list[GitHubPullRequest] = []
        for node in (data.get("recentPRs") or {}).get("nodes", []):
            pulls.append(
                GitHubPullRequest(
                    number=node["number"],
                    title=node["title"],
                    state=node["state"].lower(),
                    user_login=(node.get("author") or {}).get("login"),
                    created_at=node["createdAt"],
                    updated_at=node["updatedAt"],
                    closed_at=node.get("closedAt"),
                    merged_at=node.get("mergedAt"),
                )
            )
        return pulls

    @staticmethod
    def _parse_releases(data: dict[str, Any]) -> list[GitHubRelease]:
        releases: list[GitHubRelease] = []
        for node in (data.get("releases") or {}).get("nodes", []):
            releases.append(
                GitHubRelease(
                    tag_name=node["tagName"],
                    name=node.get("name"),
                    published_at=node.get("publishedAt"),
                    draft=node.get("isDraft", False),
                    prerelease=node.get("isPrerelease", False),
                )
            )
        return releases

    @staticmethod
    def _approximate_contributors(
        commits: list[GitHubCommit],
    ) -> list[GitHubContributor]:
        """Approximate contributors from commit authors.

        Since GraphQL doesn't expose the contributors endpoint, we count
        unique commit authors and their commit counts.
        """
        author_counts: dict[str, int] = {}
        for commit in commits:
            login = commit.author_login
            if login:
                author_counts[login] = author_counts.get(login, 0) + 1

        return [
            GitHubContributor(login=login, contributions=count)
            for login, count in sorted(
                author_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

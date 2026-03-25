"""Tests for the GitHub GraphQL client."""

import httpx
import pytest
import respx

from depwatch.common.config import Settings
from depwatch.core.github_graphql import (
    GRAPHQL_URL,
    GitHubGraphQLClient,
    GraphQLError,
)

SAMPLE_GRAPHQL_RESPONSE = {
    "data": {
        "repository": {
            "databaseId": 1234,
            "name": "flask",
            "nameWithOwner": "pallets/flask",
            "description": "A micro web framework",
            "primaryLanguage": {"name": "Python"},
            "stargazerCount": 65000,
            "forkCount": 16000,
            "isArchived": False,
            "isFork": False,
            "createdAt": "2010-04-06T11:11:11Z",
            "updatedAt": "2024-01-15T00:00:00Z",
            "pushedAt": "2024-01-14T00:00:00Z",
            "defaultBranchRef": {"name": "main"},
            "licenseInfo": {"spdxId": "BSD-3-Clause"},
            "hasWikiEnabled": False,
            "hasIssuesEnabled": True,
            "issues": {"totalCount": 42},
            "commitCount365": {"target": {"history": {"totalCount": 250}}},
            "commitCount90": {"target": {"history": {"totalCount": 45}}},
            "commitCount30": {"target": {"history": {"totalCount": 12}}},
            "recentCommits": {
                "target": {
                    "history": {
                        "nodes": [
                            {
                                "oid": "abc123",
                                "message": "Fix bug",
                                "committedDate": "2024-01-10T12:00:00Z",
                                "author": {"user": {"login": "alice"}},
                                "committer": {"user": {"login": "alice"}},
                            },
                            {
                                "oid": "def456",
                                "message": "Add feature",
                                "committedDate": "2024-01-05T10:00:00Z",
                                "author": {"user": {"login": "bob"}},
                                "committer": {"user": {"login": "bob"}},
                            },
                        ]
                    }
                }
            },
            "olderCommits": {
                "target": {
                    "history": {
                        "totalCount": 5,
                        "nodes": [
                            {
                                "oid": "abc123",
                                "message": "Fix bug",
                                "committedDate": "2024-01-10T12:00:00Z",
                                "author": {"user": {"login": "alice"}},
                                "committer": {"user": {"login": "alice"}},
                            },
                            {
                                "oid": "ghi789",
                                "message": "Old commit",
                                "committedDate": "2023-06-01T08:00:00Z",
                                "author": {"user": {"login": "alice"}},
                                "committer": {"user": None},
                            },
                        ],
                    }
                }
            },
            "recentIssues": {
                "totalCount": 150,
                "nodes": [
                    {
                        "number": 100,
                        "title": "Bug report",
                        "state": "OPEN",
                        "createdAt": "2024-01-12T00:00:00Z",
                        "updatedAt": "2024-01-13T00:00:00Z",
                        "closedAt": None,
                        "comments": {"totalCount": 3},
                        "author": {"login": "carol"},
                    },
                    {
                        "number": 99,
                        "title": "Closed issue",
                        "state": "CLOSED",
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-05T00:00:00Z",
                        "closedAt": "2024-01-05T00:00:00Z",
                        "comments": {"totalCount": 1},
                        "author": None,
                    },
                ],
            },
            "recentPRs": {
                "totalCount": 80,
                "nodes": [
                    {
                        "number": 50,
                        "title": "Fix typo",
                        "state": "MERGED",
                        "createdAt": "2024-01-08T00:00:00Z",
                        "updatedAt": "2024-01-09T00:00:00Z",
                        "closedAt": "2024-01-09T00:00:00Z",
                        "mergedAt": "2024-01-09T00:00:00Z",
                        "author": {"login": "dave"},
                    },
                ],
            },
            "releases": {
                "nodes": [
                    {
                        "tagName": "v3.0.0",
                        "name": "3.0.0",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "isDraft": False,
                        "isPrerelease": False,
                    },
                ]
            },
        }
    }
}


@pytest.fixture
def settings() -> Settings:
    return Settings(github_token="test-token")


@respx.mock
async def test_fetch_repo_data(settings: Settings) -> None:
    """Full fetch_repo_data parses all entity types correctly."""
    respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=SAMPLE_GRAPHQL_RESPONSE),
    )

    async with httpx.AsyncClient() as client:
        gql = GitHubGraphQLClient(client, settings)
        repo, commits, issues, pulls, contributors, releases, counts = await gql.fetch_repo_data(
            "pallets", "flask"
        )

    # Repo
    assert repo.name == "flask"
    assert repo.full_name == "pallets/flask"
    assert repo.stargazers_count == 65000
    assert repo.archived is False
    assert repo.language == "Python"

    # Accurate commit counts from totalCount
    assert counts.commits_30d == 12
    assert counts.commits_90d == 45
    assert counts.commits_365d == 250

    # Commits (deduplicated: abc123 appears in both recent and older)
    assert len(commits) == 3
    shas = {c.sha for c in commits}
    assert shas == {"abc123", "def456", "ghi789"}

    # Issues
    assert len(issues) == 2
    assert issues[0].state == "open"
    assert issues[1].closed_at is not None

    # PRs
    assert len(pulls) == 1
    assert pulls[0].merged_at is not None

    # Contributors (approximated from commits)
    assert len(contributors) == 2
    assert contributors[0].login == "alice"
    assert contributors[0].contributions == 2  # abc123 + ghi789

    # Releases
    assert len(releases) == 1
    assert releases[0].tag_name == "v3.0.0"


@respx.mock
async def test_graphql_error(settings: Settings) -> None:
    """GraphQL errors are raised as GraphQLError."""
    error_response = {
        "errors": [{"message": "Repository not found"}],
    }
    respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=error_response),
    )

    async with httpx.AsyncClient() as client:
        gql = GitHubGraphQLClient(client, settings)
        with pytest.raises(GraphQLError, match="Repository not found"):
            await gql.fetch_repo_data("nonexistent", "repo")


@respx.mock
async def test_repo_not_found(settings: Settings) -> None:
    """Null repository data raises GraphQLError."""
    response = {"data": {"repository": None}}
    respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=response),
    )

    async with httpx.AsyncClient() as client:
        gql = GitHubGraphQLClient(client, settings)
        with pytest.raises(GraphQLError, match="not found"):
            await gql.fetch_repo_data("ghost", "repo")


@respx.mock
async def test_auth_header(settings: Settings) -> None:
    """Authorization header is set when token is provided."""
    route = respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=SAMPLE_GRAPHQL_RESPONSE),
    )

    async with httpx.AsyncClient() as client:
        gql = GitHubGraphQLClient(client, settings)
        await gql.fetch_repo_data("pallets", "flask")

    assert route.called
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-token"


@respx.mock
async def test_null_author(settings: Settings) -> None:
    """Commits with null authors are handled gracefully."""
    response = {
        "data": {
            "repository": {
                **SAMPLE_GRAPHQL_RESPONSE["data"]["repository"],
                "recentCommits": {
                    "target": {
                        "history": {
                            "nodes": [
                                {
                                    "oid": "null_author",
                                    "message": "bot commit",
                                    "committedDate": "2024-01-10T12:00:00Z",
                                    "author": {"user": None},
                                    "committer": {"user": None},
                                }
                            ]
                        }
                    }
                },
                "olderCommits": {"target": None},
            }
        }
    }
    respx.post(GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json=response),
    )

    async with httpx.AsyncClient() as client:
        gql = GitHubGraphQLClient(client, settings)
        (
            _repo,
            commits,
            _issues,
            _pulls,
            contributors,
            _releases,
            _counts,
        ) = await gql.fetch_repo_data("pallets", "flask")

    assert len(commits) == 1
    assert commits[0].author_login is None
    # No contributors since author is None
    assert len(contributors) == 0

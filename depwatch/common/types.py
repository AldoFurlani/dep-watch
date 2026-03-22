"""Shared Pydantic models for GitHub API responses and package registry data."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# --- GitHub API response models ---


class GitHubRepo(BaseModel):
    """Core repository metadata from GET /repos/{owner}/{repo}."""

    id: int
    name: str
    full_name: str
    owner_login: str = Field(alias="owner_login")
    description: str | None = None
    language: str | None = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    watchers_count: int = 0
    archived: bool = False
    fork: bool = False
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime | None = None
    default_branch: str = "main"
    license_name: str | None = None
    has_wiki: bool = False
    has_issues: bool = True

    model_config = {"populate_by_name": True}

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GitHubRepo":
        """Parse from raw GitHub API JSON response."""
        return cls(
            id=data["id"],
            name=data["name"],
            full_name=data["full_name"],
            owner_login=data["owner"]["login"],
            description=data.get("description"),
            language=data.get("language"),
            stargazers_count=data.get("stargazers_count", 0),
            forks_count=data.get("forks_count", 0),
            open_issues_count=data.get("open_issues_count", 0),
            watchers_count=data.get("watchers_count", 0),
            archived=data.get("archived", False),
            fork=data.get("fork", False),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            pushed_at=data.get("pushed_at"),
            default_branch=data.get("default_branch", "main"),
            license_name=data.get("license", {}).get("spdx_id") if data.get("license") else None,
            has_wiki=data.get("has_wiki", False),
            has_issues=data.get("has_issues", True),
        )


class GitHubCommit(BaseModel):
    """A commit from GET /repos/{owner}/{repo}/commits."""

    sha: str
    author_login: str | None = None
    committer_login: str | None = None
    message: str = ""
    committed_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GitHubCommit":
        author = data.get("author")
        committer = data.get("committer")
        commit_info = data.get("commit", {})
        committer_info = commit_info.get("committer", {})
        return cls(
            sha=data["sha"],
            author_login=author["login"] if author else None,
            committer_login=committer["login"] if committer else None,
            message=commit_info.get("message", ""),
            committed_at=committer_info.get("date"),
        )


class GitHubIssue(BaseModel):
    """An issue from GET /repos/{owner}/{repo}/issues."""

    number: int
    title: str
    state: str  # "open" or "closed"
    user_login: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    comments: int = 0
    is_pull_request: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GitHubIssue":
        user = data.get("user")
        return cls(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            user_login=user["login"] if user else None,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            closed_at=data.get("closed_at"),
            comments=data.get("comments", 0),
            is_pull_request="pull_request" in data,
        )


class GitHubPullRequest(BaseModel):
    """A pull request from GET /repos/{owner}/{repo}/pulls."""

    number: int
    title: str
    state: str  # "open", "closed"
    user_login: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    merged_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GitHubPullRequest":
        user = data.get("user")
        return cls(
            number=data["number"],
            title=data["title"],
            state=data["state"],
            user_login=user["login"] if user else None,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            closed_at=data.get("closed_at"),
            merged_at=data.get("merged_at"),
        )


class GitHubContributor(BaseModel):
    """A contributor from GET /repos/{owner}/{repo}/contributors."""

    login: str
    contributions: int = 0

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GitHubContributor":
        return cls(
            login=data["login"],
            contributions=data.get("contributions", 0),
        )


class GitHubRelease(BaseModel):
    """A release from GET /repos/{owner}/{repo}/releases."""

    tag_name: str
    name: str | None = None
    published_at: datetime | None = None
    draft: bool = False
    prerelease: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GitHubRelease":
        return cls(
            tag_name=data["tag_name"],
            name=data.get("name"),
            published_at=data.get("published_at"),
            draft=data.get("draft", False),
            prerelease=data.get("prerelease", False),
        )


# --- Package registry models ---


class PackageMetadata(BaseModel):
    """Unified package metadata from any registry (PyPI, npm, etc.)."""

    ecosystem: str  # "pypi", "npm"
    name: str
    latest_version: str | None = None
    description: str | None = None
    repository_url: str | None = None
    maintainer_count: int = 0
    release_count: int = 0
    latest_release_date: datetime | None = None
    first_release_date: datetime | None = None
    is_deprecated: bool = False
    weekly_downloads: int | None = None
    dependent_count: int | None = None

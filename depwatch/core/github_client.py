"""Rate-limited GitHub API client."""

import asyncio
import base64
import logging
from datetime import UTC, datetime
from typing import Any

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


class GitHubRateLimitError(Exception):
    """Raised when GitHub API rate limit is exhausted."""

    def __init__(self, reset_at: int) -> None:
        self.reset_at = reset_at
        super().__init__(f"Rate limit exhausted, resets at {reset_at}")


class GitHubClient:
    """Async GitHub API client with rate limiting.

    Args:
        http_client: An httpx.AsyncClient instance (for dependency injection in tests).
        settings: Application settings.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self._http = http_client
        self._settings = settings
        self._rate_limit_remaining: int = 5000
        self._rate_limit_reset: int = 0

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self._settings.github_token:
            headers["Authorization"] = f"Bearer {self._settings.github_token}"
        return headers

    def _update_rate_limit(self, response: httpx.Response) -> None:
        """Read rate limit headers from a GitHub API response."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._rate_limit_remaining = int(remaining)
        if reset is not None:
            self._rate_limit_reset = int(reset)

    async def _wait_for_rate_limit(self) -> None:
        """Sleep until rate limit resets if we're close to exhaustion."""
        if self._rate_limit_remaining > 50:
            return
        now = int(datetime.now(tz=UTC).timestamp())
        wait_seconds = max(self._rate_limit_reset - now + 1, 1)
        logger.warning(
            "Rate limit low (%d remaining), waiting %ds",
            self._rate_limit_remaining,
            wait_seconds,
        )
        await asyncio.sleep(wait_seconds)

    async def _get(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Make a GET request to the GitHub API with rate limiting.

        Returns the parsed JSON response.
        Raises GitHubRateLimitError if rate limit is exhausted.
        Raises httpx.HTTPStatusError for other HTTP errors.
        """
        url = f"{self._settings.github_api_base_url}{endpoint}"

        await self._wait_for_rate_limit()

        response = await self._http.get(url, headers=self._headers(), params=params)
        self._update_rate_limit(response)

        if response.status_code == 403 and self._rate_limit_remaining == 0:
            raise GitHubRateLimitError(self._rate_limit_reset)

        response.raise_for_status()
        data: dict[str, Any] | list[Any] = response.json()
        return data

    async def _get_paginated(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
        max_pages: int = 5,
        per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch multiple pages from a paginated GitHub API endpoint."""
        all_items: list[dict[str, Any]] = []
        page_params = dict(params or {})
        page_params["per_page"] = str(per_page)

        for page in range(1, max_pages + 1):
            page_params["page"] = str(page)

            await self._wait_for_rate_limit()
            url = f"{self._settings.github_api_base_url}{endpoint}"
            response = await self._http.get(url, headers=self._headers(), params=page_params)
            self._update_rate_limit(response)

            if response.status_code == 403 and self._rate_limit_remaining == 0:
                raise GitHubRateLimitError(self._rate_limit_reset)
            response.raise_for_status()

            items: list[dict[str, Any]] = response.json()
            all_items.extend(items)
            if len(items) < per_page:
                break

        return all_items

    # --- Public API methods ---

    async def get_repo(self, owner: str, name: str) -> GitHubRepo:
        """Fetch repository metadata."""
        data = await self._get(f"/repos/{owner}/{name}")
        assert isinstance(data, dict)
        return GitHubRepo.from_api(data)

    async def get_commits(
        self,
        owner: str,
        name: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[GitHubCommit]:
        """Fetch commits, optionally filtered by date range."""
        params: dict[str, str] = {}
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()

        items = await self._get_paginated(
            f"/repos/{owner}/{name}/commits",
            params=params,
        )
        return [GitHubCommit.from_api(item) for item in items]

    async def get_issues(
        self,
        owner: str,
        name: str,
        state: str = "all",
        since: datetime | None = None,
    ) -> list[GitHubIssue]:
        """Fetch issues (excludes pull requests via post-filtering)."""
        params: dict[str, str] = {"state": state}
        if since:
            params["since"] = since.isoformat()

        items = await self._get_paginated(
            f"/repos/{owner}/{name}/issues",
            params=params,
        )
        issues = [GitHubIssue.from_api(item) for item in items]
        return [i for i in issues if not i.is_pull_request]

    async def get_pulls(
        self,
        owner: str,
        name: str,
        state: str = "all",
    ) -> list[GitHubPullRequest]:
        """Fetch pull requests."""
        params: dict[str, str] = {"state": state}
        items = await self._get_paginated(
            f"/repos/{owner}/{name}/pulls",
            params=params,
        )
        return [GitHubPullRequest.from_api(item) for item in items]

    async def get_contributors(self, owner: str, name: str) -> list[GitHubContributor]:
        """Fetch repository contributors."""
        items = await self._get_paginated(f"/repos/{owner}/{name}/contributors")
        return [GitHubContributor.from_api(item) for item in items]

    async def get_releases(self, owner: str, name: str) -> list[GitHubRelease]:
        """Fetch repository releases."""
        items = await self._get_paginated(f"/repos/{owner}/{name}/releases")
        return [GitHubRelease.from_api(item) for item in items]

    async def get_readme(self, owner: str, name: str) -> str | None:
        """Fetch the decoded README content. Returns None if no README exists."""
        try:
            data = await self._get(f"/repos/{owner}/{name}/readme")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        assert isinstance(data, dict)

        content = data.get("content", "")
        encoding = data.get("encoding", "base64")
        if encoding == "base64" and content:
            return base64.b64decode(content).decode("utf-8", errors="replace")
        return content or None

    @property
    def rate_limit_remaining(self) -> int:
        return self._rate_limit_remaining

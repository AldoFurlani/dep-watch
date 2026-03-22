"""npm registry client."""

import logging
from typing import Any

import httpx

from depwatch.common.config import Settings
from depwatch.common.types import PackageMetadata

logger = logging.getLogger(__name__)


class NpmClient:
    """Fetches package metadata from the npm registry.

    Args:
        http_client: An httpx.AsyncClient instance (for dependency injection in tests).
        settings: Application settings.
    """

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http_client
        self._settings = settings

    async def get_package(self, package_name: str) -> PackageMetadata | None:
        """Fetch metadata for an npm package. Returns None if the package doesn't exist."""
        url = f"{self._settings.npm_registry_url}/{package_name}"
        response = await self._http.get(url)

        if response.status_code == 404:
            logger.debug("npm package not found: %s", package_name)
            return None
        response.raise_for_status()

        data = response.json()
        return self._parse_response(package_name, data)

    @staticmethod
    def _parse_response(package_name: str, data: dict[str, Any]) -> PackageMetadata:
        dist_tags = data.get("dist-tags", {})
        latest_tag = dist_tags.get("latest", "")
        versions = data.get("versions", {})
        time_data = data.get("time", {})

        # Check if latest version is deprecated
        latest_version_data = versions.get(latest_tag, {})
        is_deprecated = "deprecated" in latest_version_data

        # Repository URL — handle both dict and string forms
        # Dict: {"type": "git", "url": "git+https://github.com/o/r.git"}
        # String: "https://github.com/o/r" or "github:o/r"
        repo = data.get("repository") or {}
        if isinstance(repo, dict):
            repo_url = repo.get("url", "")
        elif isinstance(repo, str):
            repo_url = repo
        else:
            repo_url = ""
        # Clean up git+https:// prefix and .git suffix
        if repo_url.startswith("git+"):
            repo_url = repo_url[4:]
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        # Handle GitHub shorthand: "github:owner/repo"
        if repo_url.startswith("github:"):
            repo_url = f"https://github.com/{repo_url[7:]}"

        # Maintainer count
        maintainers = data.get("maintainers", [])
        maintainer_count = max(len(maintainers), 1)

        # Release dates from the time field
        release_times = {k: v for k, v in time_data.items() if k not in ("created", "modified")}
        sorted_times = sorted(release_times.values()) if release_times else []

        return PackageMetadata(
            ecosystem="npm",
            name=package_name,
            latest_version=latest_tag or None,
            description=data.get("description"),
            repository_url=repo_url or None,
            maintainer_count=maintainer_count,
            release_count=len(versions),
            latest_release_date=sorted_times[-1] if sorted_times else None,
            first_release_date=sorted_times[0] if sorted_times else None,
            is_deprecated=is_deprecated,
        )

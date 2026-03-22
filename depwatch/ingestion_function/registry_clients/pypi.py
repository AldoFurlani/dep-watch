"""PyPI registry client."""

import logging
from datetime import datetime
from typing import Any

import httpx

from depwatch.common.config import Settings
from depwatch.common.types import PackageMetadata

logger = logging.getLogger(__name__)


class PyPIClient:
    """Fetches package metadata from the PyPI JSON API.

    Args:
        http_client: An httpx.AsyncClient instance (for dependency injection in tests).
        settings: Application settings.
    """

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http_client
        self._settings = settings

    async def get_package(self, package_name: str) -> PackageMetadata | None:
        """Fetch metadata for a PyPI package. Returns None if the package doesn't exist."""
        url = f"{self._settings.pypi_url}/pypi/{package_name}/json"
        response = await self._http.get(url)

        if response.status_code == 404:
            logger.debug("PyPI package not found: %s", package_name)
            return None
        response.raise_for_status()

        data = response.json()
        return self._parse_response(package_name, data)

    @staticmethod
    def _parse_response(package_name: str, data: dict[str, Any]) -> PackageMetadata:
        info = data.get("info", {})
        releases = data.get("releases", {})

        # Extract repository URL from project URLs
        project_urls = info.get("project_urls") or {}
        repo_url = (
            project_urls.get("Source")
            or project_urls.get("Repository")
            or project_urls.get("Source Code")
            or project_urls.get("Homepage")
        )

        # Count non-yanked releases and find date range
        release_dates: list[str] = []
        for _version, files in releases.items():
            non_yanked = [f for f in files if not f.get("yanked", False)]
            if non_yanked:
                upload_time = non_yanked[0].get("upload_time_iso_8601") or non_yanked[0].get(
                    "upload_time"
                )
                if upload_time:
                    release_dates.append(upload_time)

        release_dates.sort()

        def _parse_date(s: str) -> datetime:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        # Maintainer count from author + maintainer fields
        maintainer_count = 0
        if info.get("author"):
            maintainer_count += 1
        if info.get("maintainer") and info.get("maintainer") != info.get("author"):
            maintainer_count += 1
        maintainer_count = max(maintainer_count, 1)

        return PackageMetadata(
            ecosystem="pypi",
            name=package_name,
            latest_version=info.get("version"),
            description=info.get("summary"),
            repository_url=repo_url,
            maintainer_count=maintainer_count,
            release_count=len([r for r in releases.values() if r]),
            latest_release_date=_parse_date(release_dates[-1]) if release_dates else None,
            first_release_date=_parse_date(release_dates[0]) if release_dates else None,
            is_deprecated=bool(info.get("yanked")),
        )

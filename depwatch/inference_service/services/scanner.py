"""Scanner service — orchestrates manifest parsing → data fetching → scoring.

For each package in a manifest:
1. Look up GitHub repo via registry API (PyPI/npm)
2. Check SQLite cache for existing feature snapshot
3. If not cached, fetch data via GraphQL and compute features
4. Score with XGBoost model + SHAP explanations
5. Return ranked risk report
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from depwatch.core.feature_extractor import CountOverrides, extract_features
from depwatch.core.github_client import GitHubClient
from depwatch.core.github_graphql import GitHubGraphQLClient, GraphQLError
from depwatch.inference_service.models.schemas import (
    PackageErrorResponse,
    PackageRiskResponse,
    RiskFactorResponse,
    ScanResponse,
)
from depwatch.inference_service.services.manifest_parser import (
    Ecosystem,
    ParsedDependency,
    parse_manifest,
)
from depwatch.ingestion_function.registry_clients.npm import NpmClient
from depwatch.ingestion_function.registry_clients.pypi import PyPIClient

if TYPE_CHECKING:
    from depwatch.common.config import Settings
    from depwatch.inference_service.services.cache import SnapshotCache
    from depwatch.inference_service.services.scorer import Scorer

logger = logging.getLogger(__name__)

_GITHUB_REPO_PATTERN = re.compile(
    r"github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?(?:[/\s#?]|$)",
)


def _extract_github_owner_repo(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL."""
    match = _GITHUB_REPO_PATTERN.search(url)
    if match:
        return match.group(1), match.group(2)
    return None


class Scanner:
    """Orchestrates the full scan pipeline."""

    def __init__(
        self,
        settings: Settings,
        scorer: Scorer,
        cache: SnapshotCache,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._settings = settings
        self._scorer = scorer
        self._cache = cache
        self._http = http_client
        self._graphql = GitHubGraphQLClient(http_client, settings)
        self._rest_github = GitHubClient(http_client, settings)
        self._pypi = PyPIClient(http_client, settings)
        self._npm = NpmClient(http_client, settings)
        self._api_semaphore = asyncio.Semaphore(5)  # max concurrent API calls

    async def scan_manifest(self, filename: str, content: str) -> ScanResponse:
        """Parse a manifest and score all dependencies.

        Resolves repos first to deduplicate (e.g., react/react-dom both map
        to facebook/react), then fetches unique repos in parallel.
        """
        deps = parse_manifest(filename, content)
        logger.info("Parsed %d dependencies from %s", len(deps), filename)

        # Phase 1: Resolve all packages to GitHub repos (parallel registry lookups)
        resolve_tasks = [self._resolve_github_repo(d.name, d.ecosystem) for d in deps]
        resolved_raw = await asyncio.gather(*resolve_tasks, return_exceptions=True)

        # Build mapping: dep → (owner, repo, ecosystem) and identify unique repos
        dep_repos: list[tuple[ParsedDependency, tuple[str, str] | None]] = []
        unique_repos: dict[tuple[str, str], Ecosystem] = {}
        for dep, raw in zip(deps, resolved_raw, strict=True):
            if isinstance(raw, BaseException):
                logger.warning("Failed to resolve %s: %s", dep.name, raw)
                dep_repos.append((dep, None))
            elif raw is None:
                dep_repos.append((dep, None))
            else:
                owner_repo: tuple[str, str] = raw
                dep_repos.append((dep, owner_repo))
                repo_key = (owner_repo[0].lower(), owner_repo[1].lower())
                if repo_key not in unique_repos:
                    unique_repos[repo_key] = dep.ecosystem

        logger.info(
            "%d packages → %d unique repos",
            len(deps),
            len(unique_repos),
        )

        # Phase 2: Fetch and score unique repos in parallel
        repo_keys = list(unique_repos.keys())
        repo_ecosystems = list(unique_repos.values())
        score_coros = [
            self._score_repo(k[0], k[1], eco)
            for k, eco in zip(repo_keys, repo_ecosystems, strict=True)
        ]
        scored_raw = await asyncio.gather(*score_coros, return_exceptions=True)

        repo_results: dict[tuple[str, str], PackageRiskResponse | None] = {}
        repo_errors: dict[tuple[str, str], str] = {}
        for repo_key_i, score_result in zip(repo_keys, scored_raw, strict=True):
            if isinstance(score_result, BaseException):
                logger.warning("Failed to score %s/%s: %s", *repo_key_i, score_result)
                repo_errors[repo_key_i] = "Internal error during scoring"
            elif score_result is None:
                repo_errors[repo_key_i] = "Could not fetch GitHub data"
            else:
                repo_results[repo_key_i] = score_result

        # Phase 3: Map results back to packages
        results: list[PackageRiskResponse] = []
        errors: list[PackageErrorResponse] = []

        for dep, maybe_repo in dep_repos:
            if maybe_repo is None:
                errors.append(
                    PackageErrorResponse(
                        package=dep.name,
                        ecosystem=dep.ecosystem,
                        error="Could not find GitHub repository",
                    )
                )
                continue

            repo_key = (maybe_repo[0].lower(), maybe_repo[1].lower())

            if repo_key in repo_errors:
                errors.append(
                    PackageErrorResponse(
                        package=dep.name,
                        ecosystem=dep.ecosystem,
                        error=repo_errors[repo_key],
                    )
                )
            elif repo_key in repo_results:
                scored = repo_results[repo_key]
                if scored is None:  # pragma: no cover — guarded by dict membership
                    continue
                results.append(
                    PackageRiskResponse(
                        package=dep.name,
                        ecosystem=dep.ecosystem,
                        github_repo=scored.github_repo,
                        abandonment_probability=scored.abandonment_probability,
                        risk_level=scored.risk_level,
                        top_risk_factors=scored.top_risk_factors,
                    )
                )

        results.sort(key=lambda r: r.abandonment_probability, reverse=True)

        return ScanResponse(
            packages_scanned=len(deps),
            packages_scored=len(results),
            packages_errored=len(errors),
            results=results,
            errors=errors,
        )

    async def _score_repo(
        self,
        owner: str,
        repo: str,
        ecosystem: Ecosystem,
    ) -> PackageRiskResponse | None:
        """Fetch data, compute features, and score a single GitHub repo."""
        now = datetime.now(tz=UTC)
        month = now.strftime("%Y-%m")

        # Check cache (no semaphore needed — local I/O)
        cached = self._cache.get(owner, repo, month)
        if cached is not None:
            logger.info("Cache hit for %s/%s (%s)", owner, repo, month)
            scored = self._scorer.score(repo, ecosystem, owner, repo, cached)
            return _to_response(scored)

        # Fetch data via GraphQL + REST (throttled to avoid rate limits)
        async with self._api_semaphore:
            try:
                (
                    repo_data,
                    commits,
                    issues,
                    pulls,
                    _gql_contributors,
                    releases,
                    counts,
                ) = await self._graphql.fetch_repo_data(owner, repo)
            except GraphQLError:
                logger.exception("GraphQL error for %s/%s", owner, repo)
                return None

            # Fetch contributors via REST for accurate counts
            try:
                contributors = await self._rest_github.get_contributors(owner, repo)
            except (httpx.HTTPError, httpx.InvalidURL, KeyError, ValueError):
                logger.warning(
                    "REST contributors failed for %s/%s, using GraphQL approx",
                    owner,
                    repo,
                )
                contributors = _gql_contributors

        # Compute features with accurate counts
        features = extract_features(
            repo=repo_data,
            commits=commits,
            issues=issues,
            pulls=pulls,
            contributors=contributors,
            releases=releases,
            snapshot_date=now,
            count_overrides=CountOverrides(
                commits_30d=counts.commits_30d,
                commits_90d=counts.commits_90d,
                commits_365d=counts.commits_365d,
                issues_90d=counts.issues_90d,
            ),
        )

        # Cache and score
        self._cache.put(owner, repo, month, features)
        scored = self._scorer.score(repo, ecosystem, owner, repo, features)
        return _to_response(scored)

    async def _resolve_github_repo(
        self,
        package_name: str,
        ecosystem: Ecosystem,
    ) -> tuple[str, str] | None:
        """Look up the GitHub repo for a package via its registry."""
        if ecosystem == "go":
            # Go module paths are already GitHub paths (for github.com modules)
            result = _extract_github_owner_repo(package_name)
            if result:
                return result
            return None

        if ecosystem == "pypi":
            meta = await self._pypi.get_package(package_name)
        elif ecosystem == "npm":
            meta = await self._npm.get_package(package_name)
        else:
            return None

        if meta is None or meta.repository_url is None:
            return None

        return _extract_github_owner_repo(meta.repository_url)


def _to_response(scored: object) -> PackageRiskResponse:
    """Convert a ScoredPackage to a PackageRiskResponse."""
    from depwatch.inference_service.services.scorer import ScoredPackage

    if not isinstance(scored, ScoredPackage):
        msg = f"Expected ScoredPackage, got {type(scored).__name__}"
        raise TypeError(msg)
    return PackageRiskResponse(
        package=scored.package_name,
        ecosystem=scored.ecosystem,
        github_repo=f"{scored.owner}/{scored.repo}",
        abandonment_probability=scored.abandonment_probability,
        risk_level=scored.risk_level,
        top_risk_factors=[
            RiskFactorResponse(
                feature=rf.feature_name,
                value=round(rf.feature_value, 4),
                impact=round(rf.shap_value, 4),
                description=rf.description,
            )
            for rf in scored.top_risk_factors
        ],
    )

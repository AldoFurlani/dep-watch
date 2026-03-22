"""Tests for PyPI and npm registry clients."""

import httpx
import pytest
import respx

from depwatch.common.config import Settings
from depwatch.ingestion_function.registry_clients.npm import NpmClient
from depwatch.ingestion_function.registry_clients.pypi import PyPIClient
from tests.conftest import REGISTRY_FIXTURES, load_fixture


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def pypi_flask_fixture() -> dict:
    return load_fixture(REGISTRY_FIXTURES / "pypi_flask.json")  # type: ignore[return-value]


@pytest.fixture
def npm_express_fixture() -> dict:
    return load_fixture(REGISTRY_FIXTURES / "npm_express.json")  # type: ignore[return-value]


class TestPyPIClient:
    @respx.mock
    async def test_get_package_parses_metadata(
        self, mock_settings: Settings, pypi_flask_fixture: dict
    ) -> None:
        respx.get("https://pypi.org/pypi/flask/json").mock(
            return_value=httpx.Response(200, json=pypi_flask_fixture)
        )

        async with httpx.AsyncClient() as http:
            client = PyPIClient(http, mock_settings)
            pkg = await client.get_package("flask")

        assert pkg is not None
        assert pkg.ecosystem == "pypi"
        assert pkg.name == "flask"
        assert pkg.latest_version == "3.1.0"
        assert pkg.repository_url == "https://github.com/pallets/flask"
        assert pkg.maintainer_count == 2  # author + maintainer
        assert pkg.release_count == 5
        assert pkg.is_deprecated is False

    @respx.mock
    async def test_get_package_extracts_release_dates(
        self, mock_settings: Settings, pypi_flask_fixture: dict
    ) -> None:
        respx.get("https://pypi.org/pypi/flask/json").mock(
            return_value=httpx.Response(200, json=pypi_flask_fixture)
        )

        async with httpx.AsyncClient() as http:
            client = PyPIClient(http, mock_settings)
            pkg = await client.get_package("flask")

        assert pkg is not None
        assert pkg.first_release_date is not None
        assert pkg.latest_release_date is not None
        # First release should be before latest
        assert pkg.first_release_date < pkg.latest_release_date

    @respx.mock
    async def test_get_package_returns_none_on_404(self, mock_settings: Settings) -> None:
        respx.get("https://pypi.org/pypi/nonexistent-pkg/json").mock(
            return_value=httpx.Response(404)
        )

        async with httpx.AsyncClient() as http:
            client = PyPIClient(http, mock_settings)
            pkg = await client.get_package("nonexistent-pkg")

        assert pkg is None

    @respx.mock
    async def test_get_package_handles_missing_project_urls(self, mock_settings: Settings) -> None:
        data = {
            "info": {
                "name": "minimal-pkg",
                "version": "0.1.0",
                "summary": None,
                "author": "someone",
                "maintainer": None,
                "project_urls": None,
                "yanked": False,
            },
            "releases": {},
        }
        respx.get("https://pypi.org/pypi/minimal-pkg/json").mock(
            return_value=httpx.Response(200, json=data)
        )

        async with httpx.AsyncClient() as http:
            client = PyPIClient(http, mock_settings)
            pkg = await client.get_package("minimal-pkg")

        assert pkg is not None
        assert pkg.repository_url is None
        assert pkg.maintainer_count == 1
        assert pkg.release_count == 0


class TestNpmClient:
    @respx.mock
    async def test_get_package_parses_metadata(
        self, mock_settings: Settings, npm_express_fixture: dict
    ) -> None:
        respx.get("https://registry.npmjs.org/express").mock(
            return_value=httpx.Response(200, json=npm_express_fixture)
        )

        async with httpx.AsyncClient() as http:
            client = NpmClient(http, mock_settings)
            pkg = await client.get_package("express")

        assert pkg is not None
        assert pkg.ecosystem == "npm"
        assert pkg.name == "express"
        assert pkg.latest_version == "4.21.0"
        assert pkg.repository_url == "https://github.com/expressjs/express"
        assert pkg.maintainer_count == 2
        assert pkg.release_count == 2
        assert pkg.is_deprecated is False

    @respx.mock
    async def test_get_package_cleans_git_url(
        self, mock_settings: Settings, npm_express_fixture: dict
    ) -> None:
        respx.get("https://registry.npmjs.org/express").mock(
            return_value=httpx.Response(200, json=npm_express_fixture)
        )

        async with httpx.AsyncClient() as http:
            client = NpmClient(http, mock_settings)
            pkg = await client.get_package("express")

        assert pkg is not None
        # Should strip git+ prefix and .git suffix
        assert not pkg.repository_url.startswith("git+")  # type: ignore[union-attr]
        assert not pkg.repository_url.endswith(".git")  # type: ignore[union-attr]

    @respx.mock
    async def test_get_package_returns_none_on_404(self, mock_settings: Settings) -> None:
        respx.get("https://registry.npmjs.org/nonexistent-pkg").mock(
            return_value=httpx.Response(404)
        )

        async with httpx.AsyncClient() as http:
            client = NpmClient(http, mock_settings)
            pkg = await client.get_package("nonexistent-pkg")

        assert pkg is None

    @respx.mock
    async def test_get_package_detects_deprecated(self, mock_settings: Settings) -> None:
        data = {
            "name": "old-pkg",
            "dist-tags": {"latest": "1.0.0"},
            "versions": {"1.0.0": {"deprecated": "Use new-pkg instead"}},
            "time": {"1.0.0": "2020-01-01T00:00:00.000Z"},
            "maintainers": [{"name": "someone"}],
        }
        respx.get("https://registry.npmjs.org/old-pkg").mock(
            return_value=httpx.Response(200, json=data)
        )

        async with httpx.AsyncClient() as http:
            client = NpmClient(http, mock_settings)
            pkg = await client.get_package("old-pkg")

        assert pkg is not None
        assert pkg.is_deprecated is True

    @respx.mock
    async def test_get_package_handles_missing_repository(self, mock_settings: Settings) -> None:
        data = {
            "name": "no-repo",
            "dist-tags": {"latest": "1.0.0"},
            "versions": {"1.0.0": {}},
            "time": {},
            "maintainers": [],
        }
        respx.get("https://registry.npmjs.org/no-repo").mock(
            return_value=httpx.Response(200, json=data)
        )

        async with httpx.AsyncClient() as http:
            client = NpmClient(http, mock_settings)
            pkg = await client.get_package("no-repo")

        assert pkg is not None
        assert pkg.repository_url is None
        assert pkg.maintainer_count == 1  # minimum 1

"""Tests for depwatch.core.github_client."""

import httpx
import pytest
import respx

from depwatch.common.config import Settings
from depwatch.core.github_client import GitHubClient, GitHubRateLimitError
from tests.conftest import GITHUB_FIXTURES, load_fixture


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        github_token="test-token",
        github_api_base_url="https://api.github.com",
    )


@pytest.fixture
def repo_fixture() -> dict:
    return load_fixture(GITHUB_FIXTURES / "repo_pallets_flask.json")  # type: ignore[return-value]


@pytest.fixture
def commits_fixture() -> list:
    return load_fixture(GITHUB_FIXTURES / "commits_pallets_flask.json")  # type: ignore[return-value]


@pytest.fixture
def issues_fixture() -> list:
    return load_fixture(GITHUB_FIXTURES / "issues_pallets_flask.json")  # type: ignore[return-value]


@pytest.fixture
def pulls_fixture() -> list:
    return load_fixture(GITHUB_FIXTURES / "pulls_pallets_flask.json")  # type: ignore[return-value]


@pytest.fixture
def contributors_fixture() -> list:
    return load_fixture(GITHUB_FIXTURES / "contributors_pallets_flask.json")  # type: ignore[return-value]


@pytest.fixture
def releases_fixture() -> list:
    return load_fixture(GITHUB_FIXTURES / "releases_pallets_flask.json")  # type: ignore[return-value]


def _rate_limit_headers(remaining: int = 4999) -> dict[str, str]:
    return {
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": "9999999999",
    }


class TestGetRepo:
    @respx.mock
    async def test_get_repo_parses_response(
        self, mock_settings: Settings, repo_fixture: dict
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask").mock(
            return_value=httpx.Response(200, json=repo_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            repo = await client.get_repo("pallets", "flask")

        assert repo.name == "flask"
        assert repo.full_name == "pallets/flask"
        assert repo.owner_login == "pallets"
        assert repo.language == "Python"
        assert repo.stargazers_count == 68000
        assert repo.archived is False
        assert repo.license_name == "BSD-3-Clause"

    @respx.mock
    async def test_get_repo_sends_auth_header(
        self, mock_settings: Settings, repo_fixture: dict
    ) -> None:
        route = respx.get("https://api.github.com/repos/pallets/flask").mock(
            return_value=httpx.Response(200, json=repo_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            await client.get_repo("pallets", "flask")

        assert route.calls[0].request.headers["Authorization"] == "Bearer test-token"


class TestGetCommits:
    @respx.mock
    async def test_get_commits_parses_response(
        self, mock_settings: Settings, commits_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/commits").mock(
            return_value=httpx.Response(200, json=commits_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            commits = await client.get_commits("pallets", "flask")

        assert len(commits) == 3
        assert commits[0].sha == "abc123def456"
        assert commits[0].author_login == "davidism"
        assert commits[2].author_login is None  # null author

    @respx.mock
    async def test_get_commits_handles_null_author(
        self, mock_settings: Settings, commits_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/commits").mock(
            return_value=httpx.Response(200, json=commits_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            commits = await client.get_commits("pallets", "flask")

        null_author_commit = commits[2]
        assert null_author_commit.author_login is None
        assert null_author_commit.committer_login is None
        assert null_author_commit.message == "Update dependencies"


class TestGetIssues:
    @respx.mock
    async def test_get_issues_filters_pull_requests(
        self, mock_settings: Settings, issues_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/issues").mock(
            return_value=httpx.Response(200, json=issues_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            issues = await client.get_issues("pallets", "flask")

        # Issue #5430 has pull_request key and should be filtered out
        assert len(issues) == 2
        assert all(not i.is_pull_request for i in issues)

    @respx.mock
    async def test_get_issues_parses_dates(
        self, mock_settings: Settings, issues_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/issues").mock(
            return_value=httpx.Response(200, json=issues_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            issues = await client.get_issues("pallets", "flask")

        closed_issue = next(i for i in issues if i.state == "closed")
        assert closed_issue.closed_at is not None
        open_issue = next(i for i in issues if i.state == "open")
        assert open_issue.closed_at is None


class TestGetPulls:
    @respx.mock
    async def test_get_pulls_parses_merged(
        self, mock_settings: Settings, pulls_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/pulls").mock(
            return_value=httpx.Response(200, json=pulls_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            pulls = await client.get_pulls("pallets", "flask")

        assert len(pulls) == 2
        merged = [p for p in pulls if p.merged_at is not None]
        assert len(merged) == 1
        assert merged[0].number == 5425


class TestGetContributors:
    @respx.mock
    async def test_get_contributors_sorted_by_contributions(
        self, mock_settings: Settings, contributors_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/contributors").mock(
            return_value=httpx.Response(
                200, json=contributors_fixture, headers=_rate_limit_headers()
            )
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            contributors = await client.get_contributors("pallets", "flask")

        assert len(contributors) == 4
        assert contributors[0].login == "davidism"
        assert contributors[0].contributions == 2450


class TestGetReleases:
    @respx.mock
    async def test_get_releases_parses_response(
        self, mock_settings: Settings, releases_fixture: list
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask/releases").mock(
            return_value=httpx.Response(200, json=releases_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            releases = await client.get_releases("pallets", "flask")

        assert len(releases) == 2
        assert releases[0].tag_name == "3.1.0"
        assert releases[0].prerelease is False


class TestGetReadme:
    @respx.mock
    async def test_get_readme_decodes_base64(self, mock_settings: Settings) -> None:
        import base64

        content = base64.b64encode(b"# Flask\n\nA micro framework.").decode()
        respx.get("https://api.github.com/repos/pallets/flask/readme").mock(
            return_value=httpx.Response(
                200,
                json={"content": content, "encoding": "base64"},
                headers=_rate_limit_headers(),
            )
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            readme = await client.get_readme("pallets", "flask")

        assert readme is not None
        assert "# Flask" in readme

    @respx.mock
    async def test_get_readme_returns_none_on_404(self, mock_settings: Settings) -> None:
        respx.get("https://api.github.com/repos/owner/no-readme/readme").mock(
            return_value=httpx.Response(404, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            readme = await client.get_readme("owner", "no-readme")

        assert readme is None


class TestRateLimiting:
    @respx.mock
    async def test_raises_on_rate_limit_exhausted(
        self, mock_settings: Settings, repo_fixture: dict
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask").mock(
            return_value=httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "9999999999",
                },
            )
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            # Set remaining to low but not zero so _wait_for_rate_limit doesn't block
            client._rate_limit_remaining = 100

            with pytest.raises(GitHubRateLimitError) as exc_info:
                await client.get_repo("pallets", "flask")

            assert exc_info.value.reset_at == 9999999999

    @respx.mock
    async def test_tracks_rate_limit_remaining(
        self, mock_settings: Settings, repo_fixture: dict
    ) -> None:
        respx.get("https://api.github.com/repos/pallets/flask").mock(
            return_value=httpx.Response(
                200,
                json=repo_fixture,
                headers={
                    "X-RateLimit-Remaining": "4200",
                    "X-RateLimit-Reset": "1700000000",
                },
            )
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, mock_settings)
            await client.get_repo("pallets", "flask")

            assert client.rate_limit_remaining == 4200


class TestNoAuthToken:
    @respx.mock
    async def test_no_auth_header_when_token_empty(self, repo_fixture: dict) -> None:
        no_token_settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            github_token="",
        )
        route = respx.get("https://api.github.com/repos/pallets/flask").mock(
            return_value=httpx.Response(200, json=repo_fixture, headers=_rate_limit_headers())
        )

        async with httpx.AsyncClient() as http:
            client = GitHubClient(http, no_token_settings)
            await client.get_repo("pallets", "flask")

        assert "Authorization" not in route.calls[0].request.headers

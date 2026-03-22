"""Shared test fixtures."""

import json
from pathlib import Path

import pytest

from depwatch.common.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GITHUB_FIXTURES = FIXTURES_DIR / "github_api_responses"
REGISTRY_FIXTURES = FIXTURES_DIR / "registry_responses"


def load_fixture(path: Path) -> dict | list:
    """Load a JSON fixture file."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


@pytest.fixture
def settings() -> Settings:
    """Test settings with no real credentials."""
    return Settings(
        github_token="test-token-fake",
        github_api_base_url="https://api.github.com",
    )

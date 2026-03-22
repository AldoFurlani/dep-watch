"""Tests for depwatch.common.config."""

import pytest

from depwatch.common.config import Settings, get_settings


def test_default_settings() -> None:
    """Settings should have sensible defaults without any env vars."""
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert s.github_token == ""
    assert s.github_api_base_url == "https://api.github.com"
    assert s.pypi_url == "https://pypi.org"
    assert s.npm_registry_url == "https://registry.npmjs.org"
    assert s.log_level == "INFO"
    assert s.model_artifact_path == "artifacts/latest"


def test_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should be overridable via environment variables."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.github_token == "ghp_test123"
    assert s.log_level == "DEBUG"


def test_get_settings_returns_instance() -> None:
    """get_settings should return a Settings instance."""
    s = get_settings()
    assert isinstance(s, Settings)

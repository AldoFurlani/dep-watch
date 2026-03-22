"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DepWatch configuration. All values can be set via environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # GitHub API
    github_token: str = ""
    github_api_base_url: str = "https://api.github.com"

    # Google Cloud / BigQuery
    gcp_project_id: str = ""

    # Package registries
    pypi_url: str = "https://pypi.org"
    npm_registry_url: str = "https://registry.npmjs.org"

    # Model artifacts
    model_artifact_path: str = "artifacts/latest"

    # Snapshot cache
    cache_db_path: str = "data/cache/snapshots.db"

    # Logging
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Create a Settings instance. Reads from environment / .env file."""
    return Settings()

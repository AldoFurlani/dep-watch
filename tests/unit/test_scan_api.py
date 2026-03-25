"""Tests for the FastAPI /scan endpoint and scanner integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from depwatch.common.features import FeatureVector
from depwatch.inference_service.main import app
from depwatch.inference_service.models.schemas import (
    PackageErrorResponse,
    PackageRiskResponse,
    RiskFactorResponse,
    ScanResponse,
)
from depwatch.inference_service.routers import scan


def _make_features(**overrides: float) -> FeatureVector:
    defaults = {name: 0.0 for name in FeatureVector.model_fields}
    defaults.update(overrides)
    return FeatureVector(**defaults)


class TestHealthEndpoint:
    def test_health(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestScanEndpoint:
    def test_no_scanner_returns_503(self) -> None:
        """If scanner not initialized, return 503."""
        scan.set_scanner(None)  # type: ignore[arg-type]
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/scan",
            files={"file": ("requirements.txt", b"flask\n", "text/plain")},
        )
        assert response.status_code == 503

    def test_no_filename_returns_400(self) -> None:
        mock_scanner = MagicMock()
        scan.set_scanner(mock_scanner)
        client = TestClient(app, raise_server_exceptions=False)
        # Send file without filename
        response = client.post(
            "/scan",
            files={"file": (None, b"flask\n", "text/plain")},
        )
        # FastAPI returns 422 for validation errors (missing filename)
        assert response.status_code == 422

    def test_unsupported_manifest_returns_400(self) -> None:
        mock_scanner = AsyncMock()
        from depwatch.inference_service.services.manifest_parser import ManifestParseError

        mock_scanner.scan_manifest.side_effect = ManifestParseError("Unsupported")
        scan.set_scanner(mock_scanner)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/scan",
            files={"file": ("Gemfile", b"gem 'rails'\n", "text/plain")},
        )
        assert response.status_code == 400

    def test_successful_scan(self) -> None:
        mock_scanner = AsyncMock()
        mock_scanner.scan_manifest.return_value = ScanResponse(
            packages_scanned=1,
            packages_scored=1,
            packages_errored=0,
            results=[
                PackageRiskResponse(
                    package="flask",
                    ecosystem="pypi",
                    github_repo="pallets/flask",
                    abandonment_probability=0.15,
                    risk_level="low",
                    top_risk_factors=[
                        RiskFactorResponse(
                            feature="stars",
                            value=65000.0,
                            impact=-0.12,
                            description="Star count",
                        ),
                    ],
                ),
            ],
            errors=[],
        )
        scan.set_scanner(mock_scanner)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/scan",
            files={"file": ("requirements.txt", b"flask>=2.0\n", "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["packages_scanned"] == 1
        assert data["packages_scored"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["package"] == "flask"
        assert data["results"][0]["risk_level"] == "low"


class TestScanResponseModel:
    def test_serialization(self) -> None:
        response = ScanResponse(
            packages_scanned=2,
            packages_scored=1,
            packages_errored=1,
            results=[
                PackageRiskResponse(
                    package="flask",
                    ecosystem="pypi",
                    github_repo="pallets/flask",
                    abandonment_probability=0.42,
                    risk_level="medium",
                    top_risk_factors=[],
                ),
            ],
            errors=[
                PackageErrorResponse(
                    package="unknown-pkg",
                    ecosystem="pypi",
                    error="Could not find GitHub repository",
                ),
            ],
        )

        data = response.model_dump()
        assert data["packages_scanned"] == 2
        assert data["packages_scored"] == 1
        assert len(data["results"]) == 1
        assert len(data["errors"]) == 1
        assert data["results"][0]["abandonment_probability"] == 0.42

"""POST /scan endpoint — accepts a manifest file and returns risk scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from depwatch.inference_service.models.schemas import ScanResponse
from depwatch.inference_service.services.manifest_parser import ManifestParseError

if TYPE_CHECKING:
    from depwatch.inference_service.services.scanner import Scanner

router = APIRouter()

# Scanner instance injected at startup by main.py
_scanner: Scanner | None = None


def set_scanner(scanner: Scanner) -> None:
    """Set the scanner instance (called during app startup)."""
    global _scanner
    _scanner = scanner


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Scan a dependency manifest for abandonment risk",
    description=(
        "Upload a manifest file (requirements.txt, package.json, or go.mod) "
        "and receive a ranked risk report with abandonment probabilities "
        "and explanations for each dependency."
    ),
)
async def scan_manifest(
    file: Annotated[UploadFile, File(description="Manifest file to scan")],
) -> ScanResponse:
    """Scan a dependency manifest and return ranked risk scores."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    content_bytes = await file.read()
    content = content_bytes.decode("utf-8")

    try:
        return await _scanner.scan_manifest(file.filename, content)
    except ManifestParseError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

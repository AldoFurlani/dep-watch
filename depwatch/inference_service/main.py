"""FastAPI application for DepWatch inference service.

Usage:
    uvicorn depwatch.inference_service.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from fastapi import FastAPI

from depwatch.common.config import get_settings
from depwatch.inference_service.routers import scan
from depwatch.inference_service.services.cache import SnapshotCache
from depwatch.inference_service.services.scanner import Scanner
from depwatch.inference_service.services.scorer import Scorer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load model at startup, cleanup on shutdown."""
    settings = get_settings()

    # Load model
    scorer = Scorer(settings.model_artifact_path)
    try:
        scorer.load()
    except FileNotFoundError:
        logger.warning(
            "Model not found at %s — scoring will be unavailable. "
            "Train a model first with: python -m depwatch.model_training.train_baselines",
            settings.model_artifact_path,
        )
        scorer = Scorer(settings.model_artifact_path)  # unloaded scorer

    # Initialize cache and HTTP client
    cache = SnapshotCache(settings.cache_db_path)
    http_client = httpx.AsyncClient(timeout=30.0)

    # Wire up scanner
    scanner = Scanner(settings, scorer, cache, http_client)
    scan.set_scanner(scanner)

    logger.info("DepWatch inference service started")
    yield

    # Cleanup
    await http_client.aclose()
    cache.close()
    logger.info("DepWatch inference service stopped")


app = FastAPI(
    title="DepWatch",
    description="Dependency abandonment prediction service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(scan.router, tags=["scan"])


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

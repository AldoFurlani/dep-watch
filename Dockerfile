FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir hatchling

# Copy project metadata and source for building
COPY pyproject.toml .
COPY depwatch/ depwatch/

# Build wheel and install into a clean prefix
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy model artifact (baked into image)
COPY artifacts/latest/ artifacts/latest/

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "depwatch.inference_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

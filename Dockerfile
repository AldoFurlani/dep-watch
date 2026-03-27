FROM python:3.12-slim AS builder

WORKDIR /app

# Install into a virtual env so we can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project metadata and source for building
COPY pyproject.toml .
COPY depwatch/ depwatch/

# Install the package
RUN pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy model artifact (baked into image)
COPY artifacts/latest/ artifacts/latest/

# Run as non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "depwatch.inference_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

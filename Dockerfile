FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY depwatch/ depwatch/

# Copy model artifact (baked into image)
COPY artifacts/latest/ artifacts/latest/

EXPOSE 8000

CMD ["uvicorn", "depwatch.inference_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

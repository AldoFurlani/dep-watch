.PHONY: install lint fmt test test-integration typecheck up down migrate migrate-check run-api

install:
	pip install -e ".[dev,training]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

typecheck:
	mypy depwatch/

test:
	pytest tests/unit -v --cov=depwatch --cov-report=term-missing

test-integration:
	pytest tests/integration -v -m integration

test-all:
	pytest -v --cov=depwatch --cov-report=term-missing -m ""

up:
	docker compose up -d

down:
	docker compose down

migrate:
	alembic upgrade head

migrate-check:
	alembic check

run-api:
	uvicorn depwatch.inference_service.main:app --reload --host 0.0.0.0 --port 8000

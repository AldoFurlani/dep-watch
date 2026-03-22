.PHONY: install lint fmt test typecheck run-api

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

run-api:
	uvicorn depwatch.inference_service.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: install test lint typecheck security docker-build smoke

IMAGE ?= fashion-api

install:
	pip install -r requirements-dev.txt

test:
	coverage run -m pytest -q
	coverage report --fail-under=70

lint:
	ruff check src tests

typecheck:
	mypy src

security:
	pip-audit -r requirements.txt

docker-build:
	docker build -t $(IMAGE):local .

smoke:
	python -c "from fashion_api.api.app import app; print(app.title)"


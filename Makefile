.PHONY: install run test lint format build up down

install:
	poetry install

run:
	poetry run python -m adas

test:
	poetry run pytest tests/

lint:
	poetry run ruff check src/

format:
	poetry run black src/

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down
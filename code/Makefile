.PHONY: clean install-dev lint type-check check-code format

DIRS_WITH_CODE = src

clean:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache __pycache__

install-dev:
	python3 -m pip install --upgrade pip poetry
	poetry install
	poetry run pre-commit install

lint:
	poetry run ruff check $(DIRS_WITH_CODE)

type-check:
	poetry run mypy $(DIRS_WITH_CODE)

check-code: lint type-check

format:
	poetry run ruff check --fix $(DIRS_WITH_CODE)
	poetry run ruff format $(DIRS_WITH_CODE)

pydantic-model:
	datamodel-codegen --input .actor/input_schema.json --output $(DIRS_WITH_CODE)/input_model.py  --input-file-type jsonschema  --field-constraints


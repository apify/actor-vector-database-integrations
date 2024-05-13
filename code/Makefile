.PHONY: clean install-dev lint type-check check-code format

DIRS_WITH_CODE = src
DIRS_WITH_ACTORS = ../actors

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
	datamodel-codegen --input $(DIRS_WITH_ACTORS)/chroma/.actor/input_schema.json --output $(DIRS_WITH_CODE)/models/chroma_input_model.py  --input-file-type jsonschema  --field-constraints
	datamodel-codegen --input $(DIRS_WITH_ACTORS)/pinecone/.actor/input_schema.json --output $(DIRS_WITH_CODE)/models/pinecone_input_model.py  --input-file-type jsonschema  --field-constraints


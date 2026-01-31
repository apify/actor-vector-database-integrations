# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and development commands

```bash
# Install all dependencies (run from repo root)
make install-dev

# Linting and type checking
make lint              # Run ruff linter
make type-check        # Run mypy
make check-code        # Run both lint and type-check
make format            # Auto-fix lint issues and format code

# Testing
make test-unit                    # Run unit tests only
make test-integration             # Run integration tests (requires databases via docker-compose)
make test                         # Run all tests

# Run a single test
poetry run -C code pytest code/tests/test_utils.py::test_name -v

# Generate pydantic models from input schemas
make pydantic-model

# Start local databases for integration testing
docker compose up -d
```

## Running actors locally

```bash
export ACTOR_PATH_IN_DOCKER_CONTEXT=actors/pinecone  # or chroma, qdrant, etc.
apify run -p
```

## Git workflow and commit conventions

### Branching strategy

- `master` - Production branch, all PRs target this branch
- Feature branches should be created from `master`

### Commit messages and PR titles

All commits and PR titles must follow **[Conventional Commits](https://www.conventionalcommits.org/)** format:

```
<type>(<scope>): <description>
```

**Required elements:**
- **type**: `feat`, `fix`, `chore`, `refactor`, `docs`, `test`, etc.
- **scope**: Component or area affected (e.g., `chroma`, `pinecone`, `tests`, `ci`)
- **description**: Brief summary in imperative mood

**Breaking changes:** Append `!` after scope (e.g., `feat(api)!: ...`)

**Changelog classification** (append to end):
- _(none)_: user-facing change (default)
- `[admin]`: admin-only change
- `[internal]`: internal change (migrations, refactoring)
- `[ignore]`: non-important change (dependency updates, DX improvements)

**Examples:**
```
feat(chroma): add remote database support
fix(pinecone): handle rate limit errors
refactor(tests): extract common fixtures [internal]
chore(deps): update langchain version [ignore]
```

### Naming conventions

- **Functions & Variables**: `camelCase`
- **Classes, Types, Components**: `PascalCase`
- **Files & Folders**: `snake_case`
- **Constants** (module-level, immutable): `UPPER_SNAKE_CASE`
- **Booleans**: Prefix with `is`, `has`, or `should` (e.g., `isValid`, `hasFinished`)
- **Units**: Suffix with unit (e.g., `timeoutSeconds`, `maxRetries`)
- **Date/Time**: Suffix with `At` (e.g., `lastSeenAt`, `createdAt`)

## Architecture overview

This is a monorepo containing multiple Apify Actors for vector database integrations. All Actors share a common codebase in `code/src/` with database-specific implementations.

### Directory structure

- `actors/` - Individual Actor definitions (one per database: chroma, milvus, opensearch, pgvector, pinecone, qdrant, weaviate)
  - Each contains `.actor/actor.json` (Actor definition) and `.actor/input_schema.json` (input schema)
- `code/src/` - Shared source code for all Actors
- `code/tests/` - Test suite

### Core flow

1. `entrypoint.py` - Entry point, determines which database Actor to run based on `ACTOR_PATH_IN_DOCKER_CONTEXT`
2. `main.py:run_actor()` - Main orchestration: load dataset → compute embeddings → chunk text → update vector store
3. `vcs.py` - Vector store operations: delta updates, upserts, expired object deletion

### Key components

**Input Models** (`code/src/models/`): Auto-generated pydantic models from `actors/*/.actor/input_schema.json`. After modifying an input schema, run `make pydantic-model` to regenerate.

**Vector Store Implementations** (`code/src/vector_stores/`): Each database has its own module implementing `VectorDbBase` from `base.py`. Required methods: `get_by_item_id`, `update_last_seen_at`, `delete_by_item_id`, `delete_expired`, `delete_all`, `is_connected`, `search_by_vector`.

**Type Definitions** (`code/src/_types.py`): `ActorInputsDb` (union of all input models) and `VectorDb` (union of all database classes).

**Embeddings** (`code/src/emb.py`): Supports OpenAI and Cohere embedding providers.

### Data update strategies

- `deltaUpdates` - Only update changed data (compares checksums)
- `add` - Add all documents without checking existing
- `upsert` - Delete by item_id then add all documents

### Adding a new database integration

See README.md for step-by-step instructions. Key steps:
1. Add database to `docker-compose.yaml`
2. Add poetry group with dependencies in `code/pyproject.toml`
3. Create Actor in `actors/<name>/` with `.actor/actor.json` and `input_schema.json`
4. Generate pydantic model with `make pydantic-model`
5. Implement database class in `code/src/vector_stores/<name>.py` extending `VectorDbBase`
6. Register in `constants.py`, `entrypoint.py`, `_types.py`, and `vcs.py`
7. Add test fixture in `code/tests/conftest.py` and add to `DATABASE_FIXTURES` list

### Testing pattern

Integration tests use pytest fixtures for each database (e.g., `db_chroma`, `db_pinecone`). Tests are parameterized over `DATABASE_FIXTURES` list. Some databases (Pinecone, OpenSearch) have eventual consistency delays handled via `unit_test_wait_for_index`.

Environment variables for tests are loaded from `.env` file (e.g., `OPENAI_API_KEY`, database connection strings).

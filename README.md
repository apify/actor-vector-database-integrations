# Apify Vector Database Integrations

The Apify Vector Database Integrations facilitate the transfer of data from Apify Actors to a vector database. 
This process includes data processing, optional splitting into chunks, embedding computation, and data storage

These integrations support incremental updates, ensuring that only changed data is updated. 
This reduces unnecessary embedding computation and storage operations, making it ideal for search and retrieval augmented generation (RAG) use cases.

This repository contains Actors for different vector databases. 

## How does it work?

1. Retrieve a dataset as output from an Actor.
2. _[Optional]_ Split text data into chunks using [langchain](https://python.langchain.com).
3. _[Optional]_ Update only changed data.
4. Compute embeddings, e.g. using [OpenAI](https://platform.openai.com/docs/guides/embeddings) or [Cohere](https://cohere.com/embeddings).
5. Save data into the database.

## Vector database integrations (Actors)
- [Chroma](https://apify.com/apify/chroma-integration)
- [PGVector](https://github.com/pgvector/pgvector)
- [Pinecone](https://apify.com/apify/pinecone-integration)
- [Qdrant](https://qdrant.tech)

## Supported Vector Embeddings
- [OpenAI](https://platform.openai.com/docs/guides/embeddings)
- [Cohere](https://cohere.com/embeddings)

## How to add a new integration (an example for PG-Vector)?

1. Add database to [docker-compose.yml](docker-compose.yaml) for local testing (if the database is available in docker).

```
version: '3.8'

services:
  pgvector-container:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=apify
    ports:
      - "5432:5432"
```

1. Add postgres dependency to `pyproject.toml`:
   ```bash
   poetry add --group=pgvector "langchain_postgres"
   ```
   and mark the group pgvector as optional (in `pyproject.toml`):
   ```toml
   [tool.poetry.group.postgres]
   optional = true
   ```
   
1. Create a new actor in the `actors` directory, e.g. `actors/pgvector` and add the following files: 
   - `README.md` - the actor documentation
   - `.actor/actor.json` - the actor definition
   - `.actor/input_schema.json` - the actor input schema
   - 
1. Create a pydantic model for the actor input schema. Edit Makefile to generate the input schema from the model:
   ```bash
    datamodel-codegen --input $(DIRS_WITH_ACTORS)/pgvector/.actor/input_schema.json --output $(DIRS_WITH_CODE)/src/models/pgvector_input_model.py  --input-file-type jsonschema  --field-constraints
   ```
   and then run
   ```bash
   make pydantic-model
   ```
1. Create a new vector store in the `vector_stores` directory, e.g. `vector_stores/pgvector` and implement all functions
1. Add PGVector into `SupportedVectorStores` in the `constants.py` 
   ```python
      class SupportedVectorStores(str, enum.Enum):
          pgvector = "pgvector"
   ```

1. Add PGVectorDatabase into `entrypoint.py`
   ```python
      if actor_type == SupportedVectorStores.pgvector.value:
          await run_actor(PgvectorIntegration(**actor_input), actor_input)
   ```

1. Add `PGVectorDatabase` and `PgvectorIntegration`  into `_types.py`
   ```python
       ActorInputsDb: TypeAlias = ChromaIntegration | PgvectorIntegration | PineconeIntegration | QdrantIntegration
       VectorDb: TypeAlias = ChromaDatabase | PGVectorDatabase | PineconeDatabase | QdrantDatabase
   ```

1. Add `PGVectorDatabase` into `vector_stores/vcs.py`
   ```python
       if isinstance(actor_input, PgvectorIntegration):
           from .vector_stores.pgvector import PGVectorDatabase

           return PGVectorDatabase(actor_input, embeddings)
   ```

1. Add `PGVectorDatabase` fixture into `tests/confets.py`
   ```python
      @pytest.fixture()
      def db_pgvector(crawl_1: list[Document]) -> PGVectorDatabase:
          db = PGVectorDatabase(
              actor_input=PgvectorIntegration(
                  postgresSqlConnectionStr=os.getenv("POSTGRESQL_CONNECTION_STR"),
                  postgresCollectionName=INDEX_NAME,
                  embeddingsProvider=EmbeddingsProvider.OpenAI.value,
                  embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
                  datasetFields=["text"],
              ),
              embeddings=embeddings,
          )

          db.unit_test_wait_for_index = 0

          db.delete_all()
          # Insert initially crawled objects
          db.add_documents(documents=crawl_1, ids=[d.metadata["id"] for d in crawl_1])

          yield db

          db.delete_all()
   ```

1. Add the `db_pgvector` fixture into `tests/test_vector_stores.py`
   ```python
      DATABASE_FIXTURES = ["db_pinecone", "db_chroma", "db_qdrant", "db_pgvector"]
   ```
1. Update README.md in the `actors/pgvector` directory

1. Add the `pgvector` to the README.md in the root directory

1. Run tests
   ```bash  
   make test
   ```

1. Run the actor locally
   ```bash
   export ACTOR_PATH_IN_DOCKER_CONTEXT=actors/pgvector
   apify run -p
   ````

1. Setup Actor on Apify platform at `https://console.apify.com`

   Build configuration
   ```
   Git URL: https://github.com/apify/store-vector-db
   Branch: master
   Folder: actors/pgvector
   ```

1. Test the actor on the Apify platform
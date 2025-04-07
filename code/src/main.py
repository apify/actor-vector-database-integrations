from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from apify import Actor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .constants import DAY_IN_SECONDS
from .emb import get_embedding_provider
from .utils import add_chunk_id, add_item_checksum, get_dataset_loader
from .vcs import delete_expired_objects, get_vector_database, update_db_with_crawled_data, upsert_db_with_crawled_data

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from ._types import ActorInputsDb, VectorDb


async def run_actor(actor_input: ActorInputsDb, payload: dict) -> None:
    """Main function to run the actor.

    It loads the dataset, chunks the documents if necessary and updates the vector store with the new documents while removing the old ones.
    """

    payload = payload.get("payload", {})
    resource = payload.get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or actor_input.datasetId):
        msg = (
            "The `datasetId` is not provided. There are two ways to specify the datasetId:"
            "1. Automatic Input: If this integration is used with other Actors, such as the Website Content Crawler, the datasetId should be "
            "automatically passed in the 'payload'. Please check the `Input` payload to ensure the datasetId is included."
            "2. Manual Input: If you are running this Actor independently, you need to manually specify the 'datasetId'. "
            "You can do this by entering the dataset ID in the 'Dataset Settings' section of the Actor's input screen."
            "Please verify that one of these options is correctly configured to provide the datasetId."
        )

        Actor.log.error(msg)
        await Actor.fail(status_message=msg)
        return

    embeddings = await get_embeddings(actor_input)
    documents = await load_dataset(actor_input, dataset_id)
    documents = add_item_checksum(documents, actor_input.dataUpdatesPrimaryDatasetFields)  # type: ignore[arg-type]

    if actor_input.performChunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=actor_input.chunkSize, chunk_overlap=actor_input.chunkOverlap)
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))

    documents = add_chunk_id(documents)

    try:
        vcs_: VectorDb = await get_vector_database(actor_input, embeddings)
    except Exception as e:
        Actor.log.exception(e)
        await Actor.fail(
            status_message="Failed to connect/get database. Please ensure the following: "
            "1. Database credentials are correct and the database is configure properly. "
            "2. The vector dimension of your embedding model in the Actor input (Embedding settings -> model) matches the one set up in the database."
            f" Database error message: {e}"
        )
        return

    try:
        data_update_strategy = hasattr(actor_input, "dataUpdatesStrategy") and actor_input.dataUpdatesStrategy
        if data_update_strategy == "deltaUpdates":
            Actor.log.info("Update database with crawled data. Delta updates enabled")
            update_db_with_crawled_data(vcs_, documents)
        elif data_update_strategy == "add":
            vcs_.add_documents(documents)
            Actor.log.info("Added %s new objects to the vector store", len(documents))
        elif data_update_strategy == "upsert":
            upsert_db_with_crawled_data(vcs_, documents)
        else:
            await Actor.fail(
                status_message=f"Invalid dataUpdatesStrategy: {data_update_strategy}. "
                f"Please ensure that the configuration in the Database Settings is correct."
            )

        if actor_input.deleteExpiredObjects:
            expired_days = actor_input.expiredObjectDeletionPeriodDays or 0
            ts_expired = expired_days and int(datetime.now(timezone.utc).timestamp() - expired_days * DAY_IN_SECONDS) or 0
            Actor.log.info("Delete expired objects in the database: expired_days: %s", expired_days)
            delete_expired_objects(vcs_, ts_expired)

        await Actor.push_data([doc.dict() for doc in documents])

        if hasattr(vcs_, "close"):
            vcs_.close()

    except Exception as e:
        Actor.log.error(e)
        # I had to create a msg variable to avoid a ruff lint error S608 (SQL Injection)
        msg = (
            "Failed to update database. Please ensure the following:"
            "1. Database is configured properly."
            "2. The vector dimension of your embedding model in the Actor input (Embedding settings -> model) matches the one set up in the database."
            "Error message:"
        )
        await Actor.fail(status_message=f"{msg} {e}", exception=e)


async def get_embeddings(actor_input: ActorInputsDb) -> Embeddings:  # type: ignore[return]
    try:
        embed_provider_name = str(actor_input.embeddingsProvider)
        Actor.log.info("Get embeddings class: %s", embed_provider_name)
        embeddings = await get_embedding_provider(
            embed_provider_name,
            actor_input.embeddingsApiKey,
            actor_input.embeddingsConfig,
        )
    except Exception as e:
        Actor.log.error(e)
        await Actor.fail(status_message=f"Failed to get embeddings: {e}. Ensure that the configuration in the Embeddings Settings is correct.")
    else:
        return embeddings


async def load_dataset(actor_input: ActorInputsDb, dataset_id: str) -> list[Document]:  # type: ignore[return]
    """Load dataset from the datasetId and extract fields from the dataset."""

    # Add parameters related to chunking to every dataset item to be able to update DB when chunkSize, chunkOverlap or performChunking changes
    meta_object = actor_input.metadataObject or {}
    meta_object.update({"chunkSize": actor_input.chunkSize, "chunkOverlap": actor_input.chunkOverlap, "performChunking": actor_input.performChunking})

    # Required for checksum calculation
    # Update metadata fields with datasetFieldsToItemId for dataset loading
    meta_fields = actor_input.metadataDatasetFields or {}
    meta_fields.update({k: k for k in actor_input.dataUpdatesPrimaryDatasetFields or []})
    Actor.log.info("Load Dataset ID %s and extract fields %s", dataset_id, actor_input.datasetFields)

    try:
        dataset_loader = get_dataset_loader(
            str(dataset_id),
            fields=actor_input.datasetFields,
            meta_object=meta_object,
            meta_fields=meta_fields,
        )
        documents = dataset_loader.load()
        documents = [doc for doc in documents if doc.page_content]
        Actor.log.info("Dataset loaded, number of documents: %s", len(documents))

    except Exception as e:
        Actor.log.error(e)
        await Actor.fail(
            status_message=f"Failed to load datasetId {dataset_id} due to error: {e}. Ensure the following: "
            f"1. If running this Actor standalone, the dataset should exist. "
            f"2. If this Actor is configured with another Actor (in the integration section), the `datasetId` should be correctly passed. "
            f"3. If the problem persists, consider creating an issue."
        )
    else:
        return documents

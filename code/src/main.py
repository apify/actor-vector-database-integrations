from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from apify import Actor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .emb import get_embeddings
from .utils import DAY_IN_SECONDS, add_chunk_id, add_item_checksum, get_dataset_loader
from .vcs import get_vector_database, update_db_with_crawled_data

if TYPE_CHECKING:
    from ._types import ActorInputsDb, VectorDb


async def run_actor(actor_input: ActorInputsDb, payload: dict) -> None:
    """Main function to run the actor.

    It loads the dataset, chunks the documents if necessary and updates the vector store with the new documents while removing the old ones.
    """

    payload = payload.get("payload", {})
    resource = payload.get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or actor_input.datasetId):
        msg = (
            "The dataset ID is missing. Please ensure the following:"
            "1. It is provided in the payload when this integration is used with other Actors, such as the Website Content Crawler."
            "2. It is manually specified by entering 'datasetId' in the Actor's input screen."
        )
        Actor.log.error(msg)
        await Actor.fail(status_message=msg)
        return

    try:
        Actor.log.info("Get embeddings class: %s", actor_input.embeddingsProvider.value)  # type: ignore[union-attr]
        embeddings = await get_embeddings(
            actor_input.embeddingsProvider.value,  # type: ignore[union-attr]
            actor_input.embeddingsApiKey,
            actor_input.embeddingsConfig,
        )
    except Exception as e:
        Actor.log.error(e)
        await Actor.fail(status_message=f"Failed to get embeddings: {e}. Ensure that the configuration is correct.")
        return

    # Add parameters related to chunking to every dataset item to be able to update DB when chunkSize, chunkOverlap or performChunking changes
    meta_object = actor_input.metadataObject or {}
    meta_object.update({"chunkSize": actor_input.chunkSize, "chunkOverlap": actor_input.chunkOverlap, "performChunking": actor_input.performChunking})

    # Required for checksum calculation
    # Update metadata fields with datasetFieldsToItemId for dataset loading
    meta_fields = actor_input.metadataDatasetFields or {}
    meta_fields.update({k: k for k in actor_input.deltaUpdatesPrimaryDatasetFields or []})

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
            f"3. If the issue persists, consider creating an issue."
        )
        return

    documents = add_item_checksum(documents, actor_input.deltaUpdatesPrimaryDatasetFields)  # type: ignore[arg-type]

    if actor_input.performChunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=actor_input.chunkSize, chunk_overlap=actor_input.chunkOverlap)
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))

    documents = add_chunk_id(documents)

    try:
        vcs_: VectorDb = await get_vector_database(actor_input, embeddings)
    except Exception as e:
        Actor.log.error(e)
        await Actor.fail(
            status_message="Failed to connect/get database. Please ensure the following: "
            "1. Database credentials are correct and the database is configure properly. "
            "2. The vector dimension of your embedding model matches the one set up in the database."
            f" Database error message: {e}"
        )
        return
    try:
        if actor_input.enableDeltaUpdates:
            expired_days = actor_input.expiredObjectDeletionPeriodDays or 0
            ts_expired = expired_days and int(datetime.now(timezone.utc).timestamp() - expired_days * DAY_IN_SECONDS) or 0
            Actor.log.info("Update database with crawled data. Delta updates enabled, expired_days: %s, expired_ts %s", expired_days, ts_expired)
            update_db_with_crawled_data(vcs_, documents, ts_expired)
        else:
            await vcs_.aadd_documents(documents)
            Actor.log.info("Added %s new objects to the vector store", len(documents))

        await Actor.push_data([doc.dict() for doc in documents])
    except Exception as e:
        Actor.log.error(e)
        # I had to create a msg variable to avoid a ruff lint error S608 (SQL Injection)
        msg = (
            "Failed to update database. Please ensure the following:"
            "1. Database is configured properly."
            "2. The vector dimension of your embedding model matches the one set up in the database."
            "Error message:"
        )
        await Actor.fail(status_message=f"{msg} {e}")

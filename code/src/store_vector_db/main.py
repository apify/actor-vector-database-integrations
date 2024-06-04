from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from apify import Actor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from store_vector_db.emb import get_embeddings
from store_vector_db.utils import DAY_IN_SECONDS, add_chunk_id, add_item_checksum, get_dataset_loader
from store_vector_db.vcs import ActorInputsDb, get_vector_store, update_db_with_crawled_data

if TYPE_CHECKING:
    from store_vector_db.vcs import DB


async def run_actor(actor_input: ActorInputsDb, payload: dict) -> None:
    """Main function to run the actor.

    It loads the dataset, chunks the documents if necessary and updates the vector store with the new documents while removing the old ones.
    """

    payload = payload.get("payload", {})
    resource = payload.get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or actor_input.datasetId):
        msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
        await Actor.fail(status_message=msg)

    try:
        Actor.log.info("Get embeddings class: %s", actor_input.embeddingsProvider.value)  # type: ignore[union-attr]
        embeddings = await get_embeddings(
            actor_input.embeddingsProvider.value,  # type: ignore[union-attr]
            actor_input.embeddingsApiKey,
            actor_input.embeddingsConfig,
        )
    except Exception as e:
        msg = f"Failed to get embeddings: {e}"
        await Actor.fail(status_message=msg)
        return

    Actor.log.info("Load Dataset ID %s and extract fields %s", dataset_id, actor_input.datasetFields)
    try:
        dataset_loader = get_dataset_loader(
            str(actor_input.datasetId),
            fields=actor_input.datasetFields,
            meta_values=actor_input.metadataObject or {},
            meta_fields=actor_input.metadataDatasetFields or {},
        )
        documents = dataset_loader.load()
        documents = [doc for doc in documents if doc.page_content]
        Actor.log.info("Dataset loaded, number of documents: %s", len(documents))
    except Exception as e:
        await Actor.fail(status_message=f"Failed to load datasets: {e}")
        return

    documents = add_item_checksum(documents, actor_input.datasetKeysToItemId)  # type: ignore[arg-type]

    if actor_input.performChunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=actor_input.chunkSize, chunk_overlap=actor_input.chunkOverlap)
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))

    documents = add_chunk_id(documents)

    try:
        vcs_: DB = await get_vector_store(actor_input, embeddings)
        if actor_input.enableDeltaUpdates:
            expired_days = actor_input.expiredObjectDeletionPeriod or 0
            ts_expired = expired_days and int(datetime.now(timezone.utc).timestamp() - expired_days * DAY_IN_SECONDS) or 0
            update_db_with_crawled_data(vcs_, documents, ts_expired)
        else:
            await vcs_.aadd_documents(documents)
        await Actor.push_data([doc.dict() for doc in documents])
    except Exception as e:
        await Actor.set_status_message(f"Document update failed: {e}")
        await Actor.fail()

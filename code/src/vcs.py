from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from apify import Actor
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .models import ChromaIntegration, PgvectorIntegration, PineconeIntegration, QdrantIntegration
from .utils import get_chunks_to_delete, get_chunks_to_update

if TYPE_CHECKING:
    from langchain.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings

    from ._types import ActorInputsDb, VectorDb


async def get_vector_database(actor_input: ActorInputsDb | None, embeddings: Embeddings) -> VectorDb:
    """Get database based on the integration type."""

    if isinstance(actor_input, ChromaIntegration):
        from .vector_stores.chroma import ChromaDatabase

        return ChromaDatabase(actor_input, embeddings)

    if isinstance(actor_input, PgvectorIntegration):
        from .vector_stores.pgvector import PGVectorDatabase

        return PGVectorDatabase(actor_input, embeddings)

    if isinstance(actor_input, PineconeIntegration):
        from .vector_stores.pinecone import PineconeDatabase

        return PineconeDatabase(actor_input, embeddings)

    if isinstance(actor_input, QdrantIntegration):
        from .vector_stores.qdrant import QdrantDatabase

        return QdrantDatabase(actor_input, embeddings)

    raise ValueError("Unknown integration type")


def update_db_with_crawled_data(vector_store: VectorDb, documents: list[Document], ts_expired: int) -> None:
    """Update the database with new crawled data."""

    data_add, ids_update_last_seen, ids_del = compare_crawled_data_with_db(vector_store, documents)
    Actor.log.info("Objects: to add: %s, to update last_seen_at: %s, to delete: %s", len(data_add), len(ids_update_last_seen), len(ids_del))

    # Delete data that were updated
    if ids_del:
        vector_store.delete(ids_del)
        Actor.log.info("Deleted %s objects from the vector store where the content has changed since the last update", len(ids_del))

    # Add new data
    if data_add:
        vector_store.add_documents(data_add, ids=[d.metadata["id"] for d in data_add])
        Actor.log.info("Added %s new objects to the vector store", len(data_add))

    # Update metadata data
    if ids_update_last_seen:
        vector_store.update_last_seen_at(ids_update_last_seen)
        Actor.log.info("Updated last_seen_at metadata for %s objects", len(ids_update_last_seen))

    # Delete expired objects
    if ts_expired:
        dt = datetime.datetime.fromtimestamp(ts_expired, tz=datetime.timezone.utc)
        Actor.log.info("About to delete objects from the database that were not seen since %s (timestamp: %s)", dt, ts_expired)
        vector_store.delete_expired(ts_expired)


def compare_crawled_data_with_db(vector_store: VectorDb, data: list[Document]) -> tuple[list[Document], list[str], list[str]]:
    """Compare current crawled data with the data in the database. Return data to add, delete and update.

    New data is added
    Data that was not changed -> update metadata last_seen_at
    Data that was changed -> delete and add new
    """
    data_add = []
    ids_delete: set[str] = set()
    ids_update_last_seen: set[str] = set()

    crawled_db = {
        item_id: vector_store.search_by_vector(vector_store.dummy_vector, filter_={"item_id": item_id})
        for item_id in {d.metadata["item_id"] for d in data}
    }

    for d in data:
        if res := crawled_db.get(d.metadata["item_id"]):
            if d.metadata["checksum"] in {r.metadata["checksum"] for r in res}:
                ids_update_last_seen.update({r.metadata["id"]: r for r in res})
            else:
                ids_delete.update({r.metadata["id"]: r for r in res})
                data_add.append(d)
        else:
            data_add.append(d)

    return data_add, list(ids_update_last_seen), list(ids_delete)


async def update_db_with_crawled_data_using_internal_cache(
    vector_store: VectorStore, documents: list[Document], cache_key_name: str, cache_kv_store_name: str, expired_days: float
) -> None:
    """
    Updates the vector store with new documents and removes outdated ones.

    This function uses Apify's key-value store to handle documents. Each document, along with its metadata,
    is hashed and stored as a key in the key-value store.

    The function performs a comparison between the current set of documents and the set from the previous runs.
    It identifies new documents and those no longer present. New documents are added to the vector store, while
    documents older than the specified 'expired_days' are removed.
    """

    Actor.log.info("Load previous cache %s from the key-value store: %s", cache_key_name, cache_kv_store_name)

    kv_store = await Actor.open_key_value_store(name=cache_kv_store_name)
    previous_runs = await kv_store.get_value(cache_key_name) or {}
    previous_runs = [Document.parse_obj(doc) for doc in previous_runs]
    Actor.log.info("Previous runs contains: %s records", len(previous_runs))

    chunks_to_add, chunks_to_update = get_chunks_to_update(previous_runs, documents)
    chunks_to_delete, chunks_old_keep = get_chunks_to_delete(previous_runs, documents, expired_days=expired_days)

    Actor.log.info("Chunks to add: %s, chunks to update last_seen metadata: %s", len(chunks_to_add), len(chunks_to_update))
    Actor.log.info("Chunks to delete: %s", len(chunks_to_delete))

    if chunks_to_delete:
        vector_store.delete(ids=[x.metadata["id"] for x in chunks_to_delete if x.metadata["id"]])
        Actor.log.info("Deleted %s from database", len(chunks_to_delete))

    if chunks_to_add:
        inserted = vector_store.add_documents(chunks_to_add, ids=[x.metadata["id"] for x in chunks_to_add if x.metadata["id"]])
        Actor.log.info("Added %s documents to the vector store", len(inserted))
    else:
        Actor.log.info("No new documents to add")

    # update cache
    if current_cache := chunks_to_add + chunks_to_update + chunks_old_keep:
        await kv_store.set_value(cache_key_name, current_cache)
        Actor.log.info("Updated cache: %s in the key-value store with %s entries", cache_key_name, len(current_cache))

        Actor.log.info("Push chunked data to the unnamed output dataset")
        await Actor.push_data([doc.dict() for doc in current_cache])

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any, Iterator, TypeVar

import backoff
import chromadb
from chromadb.errors import ChromaError
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .base import BACKOFF_MAX_TIME_DELETE_SECONDS, BACKOFF_MAX_TIME_SECONDS, VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import ChromaIntegration

BATCH_SIZE = 300  # Chroma's default (max) size, number of documents to insert in a single request.

T = TypeVar("T")


def batch(seq: list[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class ChromaDatabase(Chroma, VectorDbBase):
    def __init__(self, actor_input: ChromaIntegration, embeddings: Embeddings) -> None:
        # Create HttpClient using partial to handle optional parameters
        client_factory = partial(
            chromadb.HttpClient,
            host=actor_input.chromaClientHost,
            ssl=actor_input.chromaClientSsl or False,
        )
        if actor_input.chromaClientPort is not None:
            client_factory = partial(client_factory, port=actor_input.chromaClientPort)
        if actor_input.chromaTenant:
            client_factory = partial(client_factory, tenant=actor_input.chromaTenant)
        if actor_input.chromaDatabase:
            client_factory = partial(client_factory, database=actor_input.chromaDatabase)
        if actor_input.chromaApiToken:
            client_factory = partial(client_factory, headers={"x-chroma-token": actor_input.chromaApiToken})
        client = client_factory()
        collection_name = actor_input.chromaCollectionName
        super().__init__(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        self.client = client
        self.index = self.client.get_or_create_collection(collection_name)
        self._dummy_vector: list[float] = []
        self.batch_size = actor_input.chromaBatchSize or BATCH_SIZE

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        if self.client.heartbeat() <= 1:
            return False
        return True

    @backoff.on_exception(backoff.expo, ChromaError, max_time=BACKOFF_MAX_TIME_SECONDS)
    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get documents by item_id."""

        results = self.index.get(where={"item_id": item_id}, include=["metadatas"])
        if (ids := results.get("ids")) and (metadata := results.get("metadatas")):
            return [Document(page_content="", metadata={**m, "chunk_id": _id}) for _id, m in zip(ids, metadata)]
        return []

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the index.

        We need to batch documents to avoid exceeding the maximum request size.
        Chroma limits the number of records we can insert in a single request to keep the payload small.
        """
        inserted_ids: list[str] = []
        batch_size = kwargs.pop("batch_size", self.batch_size)

        for docs_batch in batch(documents, batch_size):
            ids = [str(doc.metadata["chunk_id"]) for doc in docs_batch]
            batch_kwargs = {**kwargs, "ids": ids}
            inserted_ids.extend(super().add_documents(docs_batch, **batch_kwargs))
        return inserted_ids

    @backoff.on_exception(backoff.expo, ChromaError, max_time=BACKOFF_MAX_TIME_SECONDS)
    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database.

        Large updates are split into batches (self.batch_size) to avoid oversized requests.
        """
        if not ids:
            return

        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        batch_size = self.batch_size
        for ids_batch in batch(ids, batch_size):
            self.index.update(ids=ids_batch, metadatas=[{"last_seen_at": last_seen_at} for _ in ids_batch])

    @backoff.on_exception(backoff.expo, ChromaError, max_time=BACKOFF_MAX_TIME_DELETE_SECONDS)
    def delete_expired(self, expired_ts: int) -> None:
        """Delete expired objects."""
        self.index.delete(where={"last_seen_at": {"$lt": expired_ts}})  # type: ignore[dict-item]

    def delete_by_item_id(self, item_id: str) -> None:
        self.index.delete(where={"item_id": {"$eq": item_id}})  # type: ignore[dict-item]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """Delete objects by ids.

        Delete the object in batches to avoid exceeding the maximum request size.
        """
        if not ids:
            return

        for ids_batch in batch(ids, self.batch_size):
            self.index.delete(ids=ids_batch)
            super().delete(ids=ids_batch, **kwargs)

    def delete_all(self) -> None:
        """Delete all objects.

        Delete the object in batches to avoid exceeding the maximum request size.
        """
        r = self.index.get()
        if r["ids"]:
            for ids_batch in batch(r["ids"], self.batch_size):
                self.index.delete(ids=ids_batch)

    def search_by_vector(self, vector: list[float], k: int = 1_000_000, filter_: dict | None = None) -> list[Document]:
        """Search by vector."""
        return self.similarity_search_by_vector(vector, k=k, filter=filter_)

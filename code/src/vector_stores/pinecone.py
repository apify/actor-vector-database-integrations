from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import backoff
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]
from pinecone.exceptions import PineconeApiException  # type: ignore[import-untyped]

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import PineconeIntegration

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"


class PineconeDatabase(PineconeVectorStore, VectorDbBase):
    def __init__(self, actor_input: PineconeIntegration, embeddings: Embeddings) -> None:
        self.client = PineconeClient(api_key=actor_input.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
        self.index = self.client.Index(actor_input.pineconeIndexName)
        self.namespace = actor_input.pineconeIndexNamespace or None
        super().__init__(index=self.index, embedding=embeddings, namespace=self.namespace)
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        raise NotImplementedError

    @backoff.on_exception(backoff.expo, PineconeApiException, max_time=60)
    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get object by item_id.

        Pinecone does not support to get objects with filter on metadata. Hence, we need to do similarity search
        """
        results = self.index.query(
            vector=self.dummy_vector, top_k=10_000, filter={"item_id": item_id}, include_metadata=True, namespace=self.namespace
        )
        return [Document(page_content="", metadata=d["metadata"] | {"chunk_id": d["id"]}) for d in results["matches"]]

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        for _id in ids:
            self.index.update(id=_id, set_metadata={"last_seen_at": last_seen_at}, namespace=self.namespace)

    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired."""

        res = self.search_by_vector(self.dummy_vector, filter_={"last_seen_at": {"$lt": expired_ts}})
        ids = [d.metadata.get("id") or d.metadata.get("chunk_id", "") for d in res]
        ids = [_id for _id in ids if _id]
        self.delete(ids=ids)

    def delete_all(self) -> None:
        """Delete all objects from the index in the namespace that the database was initialized.

        First, get all object and then delete them.
        We can use delete_all flag but that is raising 404 exception if namespace is not found.
        Furthermore, the namespace parameter is not exposed in the function signature (the signature would collide with other databases).
        """
        if r := list(self.index.list(prefix="", namespace=self.namespace)):
            self.delete(ids=r)

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: dict | None = None) -> list[Document]:
        """Search by vector and return the results."""
        res = self.similarity_search_by_vector_with_score(vector, k=k, filter=filter_, namespace=self.namespace)
        return [r for r, _ in res]

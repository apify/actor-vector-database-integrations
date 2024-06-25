from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from ..models.pinecone_input_model import PineconeIntegration

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"


class PineconeDatabase(PineconeVectorStore, VectorDbBase):
    def __init__(self, actor_input: PineconeIntegration, embeddings: Embeddings) -> None:
        self.client = PineconeClient(api_key=actor_input.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
        super().__init__(index=self.client.Index(actor_input.pineconeIndexName), embedding=embeddings)
        self.index = self.client.Index(actor_input.pineconeIndexName)
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        raise NotImplementedError

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        for _id in ids:
            self.index.update(id=_id, set_metadata={"last_seen_at": last_seen_at})

    def delete_expired(self, expired_ts: int) -> None:
        res = self.search_by_vector(self.dummy_vector, filter_={"last_seen_at": {"$lt": expired_ts}})
        self.delete(ids=[d.metadata["id"] for d in res])

    def delete_all(self) -> None:
        # Fist, get all object and then delete them
        # We can use delete_all flag but that is raising 404 exception if namespace is not found
        if r := list(self.index.list(prefix="")):
            self.delete(ids=r)

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: dict | None = None) -> list[Document]:
        res = self.similarity_search_by_vector_with_score(vector, k=k, filter=filter_)
        return [r for r, _ in res]

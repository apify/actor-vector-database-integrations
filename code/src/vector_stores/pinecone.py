from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]

from .base import FailedToConnectToDatabaseError, VectorDbBase

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from ..models.pinecone_input_model import PineconeIntegration

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"


class PineconeDatabase(PineconeVectorStore, VectorDbBase):
    def __init__(self, actor_input: PineconeIntegration, embeddings: Embeddings) -> None:
        try:
            self.client = PineconeClient(api_key=actor_input.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
            super().__init__(index=self.client.Index(actor_input.pineconeIndexName), embedding=embeddings)
            self.index = self.client.Index(actor_input.pineconeIndexName)
            self.dummy_vector = embeddings.embed_query("dummy")
        except Exception as e:
            raise FailedToConnectToDatabaseError("Failed to connect to Pinecone") from e

    async def is_connected(self) -> bool:
        raise NotImplementedError

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        for _id in ids:
            self.index.update(id=_id, set_metadata={"last_seen_at": last_seen_at})

    def delete_expired(self, expired_ts: int) -> None:
        res = self.search_by_vector(self.dummy_vector, filter_={"last_seen_at": {"$lt": expired_ts}})
        self.delete(ids=[d.metadata["id"] for d in res])

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: dict | None = None) -> list[Document]:
        res = self.similarity_search_by_vector_with_score(vector, k=k, filter=filter_)
        return [r for r, _ in res]

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]

from store_vector_db.constants import PINECONE_SOURCE_TAG
from store_vector_db.vector_stores.base import FailedToConnectToDatabaseError, VectorDatabaseHelperClass

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from store_vector_db.models.pinecone_input_model import PineconeIntegration


class PineconeDatabase(PineconeVectorStore, VectorDatabaseHelperClass):
    def __init__(self, actor_input: PineconeIntegration, embeddings: Embeddings) -> None:

        try:
            self.client = PineconeClient(api_key=actor_input.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
            super().__init__(index=self.client.Index(actor_input.pineconeIndexName), embedding=embeddings)
            self.index = self.client.Index(actor_input.pineconeIndexName)
            self.dummy_vector = embeddings.embed_query("dummy")
        except Exception as e:
            raise FailedToConnectToDatabaseError("Failed to connect to chroma") from e

    async def is_connected(self) -> bool:
        raise NotImplementedError

    def update_metadata(self, data: list[Document]) -> None:
        for d in data:
            self.index.update(id=d.metadata["id"], set_metadata={"updated_at": d.metadata["updated_at"]})

    def delete_orphaned(self, ts_orphaned: int) -> None:
        res = self.search_by_vector(self.dummy_vector, filter_={"updated_at": {"$lt": ts_orphaned}})
        self.delete(ids=[d.metadata["id"] for d in res])

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: dict | None = None) -> list[Document]:
        res = self.similarity_search_by_vector_with_score(vector, k=k, filter=filter_)
        return [r for r, _ in res]

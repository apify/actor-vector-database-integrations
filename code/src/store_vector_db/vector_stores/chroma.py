from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from langchain_chroma import Chroma

from store_vector_db.vector_stores.base import FailedToConnectToDatabaseError, VectorDatabaseHelperClass

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from store_vector_db.models.chroma_input_model import ChromaIntegration


class ChromaDatabase(Chroma, VectorDatabaseHelperClass):
    def __init__(self, actor_input: ChromaIntegration, embeddings: Embeddings) -> None:

        settings = None
        if auth := actor_input.chromaServerAuthCredentials:
            settings = chromadb.config.Settings(
                chroma_client_auth_credentials=auth,
                chroma_client_auth_provider=actor_input.chromaClientAuthProvider,
            )
        client = chromadb.HttpClient(
            host=actor_input.chromaClientHost,
            port=actor_input.chromaClientPort or 8000,
            ssl=actor_input.chromaClientSsl or False,
            settings=settings,
        )
        collection_name = actor_input.chromaCollectionName or "chroma"
        super().__init__(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        self.client = client
        self.index = self.client.get_collection(collection_name)
        self.dummy_vector = embeddings.embed_query("dummy")

    async def is_connected(self) -> bool:
        if self.client.heartbeat() <= 1:
            raise FailedToConnectToDatabaseError("ChromaDB is not reachable")
        return True

    def update_metadata(self, data: list[Document]) -> None:
        for d in data:
            self.index.update(ids=[d.metadata["id"]], metadatas=[{"updated_at": d.metadata["updated_at"]}])

    def delete_orphaned(self, ts_orphaned: int) -> None:
        self.index.delete(where={"updated_at": {"$lt": ts_orphaned}})  # type: ignore[dict-item]

    def search_by_vector(self, vector: list[float], k: int = 1_000_000, filter_: dict | None = None) -> list[Document]:
        return self.similarity_search_by_vector(vector, k=k, filter=filter_)

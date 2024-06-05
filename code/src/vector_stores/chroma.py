from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import chromadb
from langchain_chroma import Chroma

from .base import FailedToConnectToDatabaseError, VectorDbBase

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from ..models.chroma_input_model import ChromaIntegration


class ChromaDatabase(Chroma, VectorDbBase):
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

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        for _id in ids:
            self.index.update(ids=_id, metadatas=[{"last_seen_at": last_seen_at}])

    def delete_expired(self, ts_expired: int) -> None:
        self.index.delete(where={"last_seen_at": {"$lt": ts_expired}})  # type: ignore[dict-item]

    def search_by_vector(self, vector: list[float], k: int = 1_000_000, filter_: dict | None = None) -> list[Document]:
        return self.similarity_search_by_vector(vector, k=k, filter=filter_)

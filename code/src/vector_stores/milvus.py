from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_milvus.vectorstores import Milvus
from pymilvus import MilvusClient  # type: ignore
from pymilvus.exceptions import DescribeCollectionException  # type: ignore

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import MilvusIntegration


class MilvusDatabase(Milvus, VectorDbBase):
    def __init__(self, actor_input: MilvusIntegration, embeddings: Embeddings) -> None:
        self.collection_name = actor_input.milvusCollectionName

        connection_args = {"uri": actor_input.milvusUrl, "token": actor_input.milvusApiKey}

        if actor_input.milvusUser and actor_input.milvusPassword:
            connection_args |= {"user": actor_input.milvusUser, "password": actor_input.milvusPassword}

        super().__init__(connection_args=connection_args, embedding_function=embeddings, collection_name=self.collection_name)
        self.client = MilvusClient(**connection_args)
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        raise NotImplementedError

    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get object by item_id."""

        if not item_id:
            return []

        try:
            filter_ = f"item_id == '{item_id}'"
            res = self.client.query(collection_name=self.collection_name, filter=filter_, output_fields=["chunk_id", "item_id", "checksum"])
        except DescribeCollectionException:
            return []

        return [Document(page_content="", metadata=o) for o in res]

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())

        data = self.client.get(collection_name=self.collection_name, ids=ids)
        for d in data:
            d["last_seen_at"] = last_seen_at
        self.client.upsert(collection_name=self.collection_name, data=data)

    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired."""
        self.client.delete(collection_name=self.collection_name, filter=f"last_seen_at < {expired_ts}")

    def get(self, id_: str) -> Any:
        """Get a document by id from the database.

        Used only for testing purposes.
        """
        return self.client.get(collection_name=self.collection_name, ids=[id_])

    def get_all_ids(self) -> list[str]:
        """Get all document ids from the database.

        Used only for testing purposes.
        """
        res = self.client.query(collection_name=self.collection_name, filter="", output_fields=["pk"])
        return [str(o["pk"]) for o in res]

    def delete_all(self) -> None:
        """Delete all documents from the database.

        Used only for testing purposes.
        """
        try:
            res = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["chunk_id"],
                limit=16_384,
            )
            if ids := [r.get("chunk_id") for r in res if r.get("chunk_id")]:
                self.client.delete(collection_name=self.collection_name, ids=ids)
        except DescribeCollectionException:
            return

    def search_by_vector(self, vector: list[float], k: int = 100_000, filter_: str | None = None) -> list[Document]:  # type: ignore
        return self.similarity_search_by_vector(embedding=vector, k=k, expr=filter_)

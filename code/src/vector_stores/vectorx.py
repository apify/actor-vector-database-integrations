from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from vecx import vectorx  # type: ignore
from vecx_langchain import VectorXVectorStore  # type: ignore

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import VectorxIntegration


class VectorxDatabase(VectorXVectorStore, VectorDbBase):
    def __init__(self, actor_input: VectorxIntegration, embeddings: Embeddings) -> None:
        index_name = actor_input.vectorxIndexName
        vx = vectorx.VectorX(actor_input.vectorxToken)
        self.index = vx.get_index(name=index_name, key=actor_input.vectorxKey)

        super().__init__(vectorx_index=self.index, embedding=embeddings, text_key="text")
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

        res = self.index.query_with_filter(vector=self.dummy_vector, top_k=1_000, filter={"source": item_id}, include_vectors=False)
        return [Document(page_content="", metadata=o["meta"]) for o in res]

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

        raise NotImplementedError
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        self.index.upsert()

        data = self.client.get(collection_name=self.collection_name, ids=ids)
        for d in data:
            d["last_seen_at"] = last_seen_at
        self.client.upsert(collection_name=self.collection_name, data=data)

    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired."""

        raise NotImplementedError
        self.index.delete_with_filter(filter={"last_seen_at": {"$lt": expired_ts}})

    def get(self, id_: str) -> Any:
        """Get a document by id from the database.

        Used only for testing purposes.
        """
        raise NotImplementedError
        return self.client.get(collection_name=self.collection_name, ids=[id_])

    def get_all_ids(self) -> list[str]:
        """Get all document ids from the database.

        Used only for testing purposes.
        """
        raise NotImplementedError
        res = self.client.query(collection_name=self.collection_name, filter="", output_fields=["pk"])
        return [str(o["pk"]) for o in res]

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the database."""

        for d in documents:
            # add source metadata attribute
            if "source" not in d.metadata:
                d.metadata["source"] = d.metadata["item_id"]

        return super().add_documents(documents=documents, **kwargs)

    def delete_all(self) -> None:
        """Delete all documents from the database.

        Used only for testing purposes.
        """
        self.index.delete_with_filter(filter={})

    def delete(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        """Delete a document from the database.

        Because of the vecx_langchain API is not implemented yet, we need to use the vecx API.
        """
        if ids:
            for _id in ids:
                self.index.delete_vector(_id)
        else:
            super().delete(ids=ids, filter=filter, **kwargs)

    def search_by_vector(self, vector: list[float], k: int = 100_000, filter_: str | None = None) -> list[Document]:  # type: ignore
        res = self.similarity_search_by_vector_with_score(vector, k=k, filter=filter_)
        return [r for r, _ in res]

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import backoff
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]
from pinecone.exceptions import PineconeApiException  # type: ignore[import-untyped]

from .base import BACKOFF_MAX_TIME_DELETE_SECONDS, BACKOFF_MAX_TIME_SECONDS, VectorDbBase

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
        self.use_id_prefix = actor_input.usePineconeIdPrefix
        self.embedding_batch_size = actor_input.embeddingBatchSize
        super().__init__(index=self.index, embedding=embeddings, namespace=self.namespace)
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        raise NotImplementedError

    def get_by_id(self, id_: str) -> Document:
        """Get a document by id from the database.

        Used only for testing purposes.
        """
        if result := self.index.fetch(ids=[id_], namespace=self.namespace):
            r = result["vectors"][id_]
            return Document(page_content="", metadata=r["metadata"])
        return Document(page_content="", metadata={})

    def create_prefix_id_from_item_id_chunk_id(self, doc_: Document) -> str:
        if self.use_id_prefix and "#" not in doc_.metadata["chunk_id"]:
            return f"{doc_.metadata['item_id']}#{doc_.metadata['chunk_id']}"

        return doc_.metadata["chunk_id"] or ""

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """Add documents to the index.

        Allows to use Pinecone id prefix and embedding chunk size.
        """
        if not kwargs.get("embedding_chunk_size"):
            kwargs["embedding_chunk_size"] = self.embedding_batch_size

        if (ids := kwargs.get("ids")) and self.use_id_prefix:
            # do not change id of the original document
            documents = copy.deepcopy(documents)
            for doc in documents:
                doc.metadata["chunk_id"] = self.create_prefix_id_from_item_id_chunk_id(doc)
            kwargs["ids"] = [doc.metadata["chunk_id"] for doc, _id in zip(documents, ids)]

        return super().add_documents(documents, **kwargs)

    def count(self) -> int | None:
        result = self.index.describe_index_stats(namespace=self.namespace)
        return result.get("total_vector_count", 0) or 0

    @backoff.on_exception(backoff.expo, PineconeApiException, max_time=BACKOFF_MAX_TIME_SECONDS)
    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get object by item_id.

        Pinecone does not support to get objects with filter on metadata. Hence, we need to do similarity search
        """
        if self.use_id_prefix:
            ids_ = []
            prefix = f"{item_id}#" if "#" not in item_id else item_id
            for _ids in self.index.list(prefix=prefix, namespace=self.namespace):
                ids_.extend(_ids)
            if ids_:
                results = self.index.fetch(ids=ids_, namespace=self.namespace)
                return [Document(page_content="", metadata=results["vectors"][_v]["metadata"]) for _v in results["vectors"]]
            return []

        results = self.index.query(
            vector=self.dummy_vector, top_k=10_000, filter={"item_id": item_id}, include_metadata=True, namespace=self.namespace
        )
        return [Document(page_content="", metadata=d["metadata"] | {"chunk_id": d["id"]}) for d in results["matches"]]

    @backoff.on_exception(backoff.expo, PineconeApiException, max_time=BACKOFF_MAX_TIME_SECONDS)
    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        for _id in ids:
            self.index.update(id=_id, set_metadata={"last_seen_at": last_seen_at}, namespace=self.namespace)

    @backoff.on_exception(backoff.expo, PineconeApiException, max_time=BACKOFF_MAX_TIME_DELETE_SECONDS)
    def delete_by_item_id(self, item_id: str) -> None:
        """Delete objects by item_id."""
        ids = []
        if self.use_id_prefix:
            prefix = f"{item_id}#" if "#" not in item_id else item_id
            for ids in self.index.list(prefix=prefix, namespace=self.namespace):
                ids.extend(ids)
        else:
            results = self.index.query(
                vector=self.dummy_vector, top_k=10_000, filter={"item_id": item_id}, include_metadata=True, namespace=self.namespace
            )
            ids = [r.get("id") or r.get("chunk_id") for r in results["matches"]]

        if ids:
            self.delete(ids=ids, namespace=self.namespace)

    @backoff.on_exception(backoff.expo, PineconeApiException, max_time=BACKOFF_MAX_TIME_DELETE_SECONDS)
    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired."""

        res = self.search_by_vector(self.dummy_vector, filter_={"last_seen_at": {"$lt": expired_ts}})
        ids = [d.metadata.get("id") or d.metadata.get("chunk_id", "") for d in res]
        ids = [_id for _id in ids if _id]
        self.delete(ids=ids, namespace=self.namespace)

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

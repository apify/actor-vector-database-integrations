from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import weaviate
from apify import Actor
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import WeaviateIntegration


class WeaviateDatabase(WeaviateVectorStore, VectorDbBase):
    def __init__(self, actor_input: WeaviateIntegration, embeddings: Embeddings) -> None:
        self.collection_name = actor_input.weaviateCollectionName
        self.text_key = "text"
        auth_ = weaviate.auth.AuthApiKey(actor_input.weaviateApiKey) if actor_input.weaviateApiKey else None

        if "localhost" in actor_input.weaviateUrl:
            self.client = weaviate.connect_to_local()
        else:
            self.client = weaviate.connect_to_wcs(cluster_url=actor_input.weaviateUrl, auth_credentials=auth_)

        super().__init__(client=self.client, index_name=self.collection_name, text_key=self.text_key, embedding=embeddings)
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        return self.client.is_connected()

    def close(self) -> None:
        self.client.close()

    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get object by item_id."""

        if not item_id:
            return []

        try:
            collection = self.client.collections.get(name=self.collection_name)
            response = collection.query.fetch_objects(filters=Filter.by_property("item_id").equal(item_id), limit=10_000)
        except weaviate.exceptions.WeaviateQueryError as e:
            Actor.log.warning(f"Query to Weaviate database failed. It might happen when the collection is empty. Error: {e}")
            return []

        return [Document(page_content="", metadata=dict(o.properties) | {"chunk_id": str(o.uuid)}) for o in response.objects]

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())

        collection = self.client.collections.get(name=self.collection_name)
        for _id in ids:
            collection.data.update(uuid=_id, properties={"last_seen_at": last_seen_at})

    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired."""

        collection = self.client.collections.get(name=self.collection_name)
        collection.data.delete_many(Filter.by_property("last_seen_at").less_than(expired_ts), verbose=True)

    def get(self, id_: str) -> Any:
        """Get a document by id from the database.

        Used only for testing purposes.
        """
        collection = self.client.collections.get(name=self.collection_name)
        return collection.query.fetch_object_by_id(id_, return_properties=["item_id", "checksum"])

    def get_all_ids(self) -> list[str]:
        """Get all document ids from the database.

        Used only for testing purposes.
        """
        collection = self.client.collections.get(name=self.collection_name)
        res = collection.query.fetch_objects(limit=100_000)
        return [str(o.uuid) for o in res.objects]

    def delete_all(self) -> None:
        """Delete all documents from the database.

        Used only for testing purposes.
        """
        coll = self.client.collections.get(name=self.collection_name)
        if ids := [str(o.uuid) for o in coll.iterator()]:
            self.delete(ids=ids)

    def search_by_vector(self, vector: list[float], k: int = 100_000, filter_: dict | None = None) -> list[Document]:
        if filter_:
            raise NotImplementedError("Filtering is not supported in langchain-weaviate .")

        res = self.similarity_search(query="not-used", vector=vector, k=k)
        for d in res:
            d.metadata["chunk_id"] = str(d.metadata["chunk_id"])
        return res

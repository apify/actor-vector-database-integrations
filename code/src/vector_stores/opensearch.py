from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk
from requests_aws4auth import AWS4Auth  # type: ignore

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import OpensearchIntegration

MAX_SIZE = 10_000


class OpenSearchDatabase(OpenSearchVectorSearch, VectorDbBase):
    def __init__(self, actor_input: OpensearchIntegration, embeddings: Embeddings) -> None:
        self.index_name = actor_input.openSearchIndexName
        name = actor_input.awsServiceName or ""
        name = name if isinstance(name, str) else name.value
        self.service_name = name

        if actor_input.useAWS4Auth:
            awsauth = AWS4Auth(actor_input.awsAccessKeyId, actor_input.awsSecretAccessKey, actor_input.awsRegion, self.service_name)
        else:
            awsauth = None

        super().__init__(
            connection_class=RequestsHttpConnection,
            embedding_function=embeddings,
            http_auth=awsauth,
            http_compress=True,
            index_name=self.index_name,
            opensearch_url=actor_input.openSearchUrl,
            use_ssl=actor_input.useSsl,
            verify_certs=actor_input.verifyCerts,
        )
        self.client: OpenSearch = self.client  # for type hinting across the class
        self._dummy_vector: list[float] = []

        if not self.index_exists(self.index_name):
            if actor_input.autoCreateIndex:
                v = self.dummy_vector
                self.create_index(dimension=len(v), index_name=self.index_name)
                # extra wait time for the index to be created
                time.sleep(5)
            else:
                raise ValueError(
                    f"Index '{self.index_name}' does not exist. Please create it first or enable the 'autoCreateIndex' "
                    f"option in the OpenSearch settings."
                )

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        raise NotImplementedError

    def get_by_id(self, id_: str) -> Document | None:
        """Get a document by id from the database.

        Used only for testing purposes.
        """
        if r := self.client.get(index=self.index_name, id=id_).get("_source"):
            return Document(page_content=r["text"], metadata=r["metadata"])
        return None

    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get object by item_id."""

        if not item_id:
            return []

        # noinspection PyBroadException
        try:
            res = self.client.search(
                index=self.index_name,
                body={"query": {"term": {"metadata.item_id": item_id}}, "size": MAX_SIZE},
                params={"_source_excludes": "vector_field"},
            )
        except Exception:
            return []

        if not (res := res.get("hits", {}).get("hits")):
            return []

        # OpenSearch creates a custom _id for each document, we need to return this _id in the metadata
        return [Document(page_content="", metadata={"id": o["_id"], **o["_source"]["metadata"]}) for o in res]

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database.

        Improvement: We can use bulk update to update multiple documents in a single request
        """
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        # Prepare the bulk update actions
        actions = [
            {
                "_op_type": "update",
                "_index": self.index_name,
                "_id": _id,
                "doc": {"metadata": {"last_seen_at": last_seen_at}},
            }
            for _id in ids
        ]

        # Execute the bulk update
        bulk(self.client, actions)

    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired.

        Note that delete_by_query is not working for Opensearch serverless.
        We need to search for the documents first and then delete them.
        """
        res = self.client.search(index=self.index_name, body={"query": {"range": {"metadata.last_seen_at": {"lt": expired_ts}}}, "size": MAX_SIZE})
        if not (hits := res.get("hits", {}).get("hits")):
            return

        # delete the expired documents
        self.delete(ids=[doc["_id"] for doc in hits])

    def get_all_ids(self) -> list[str]:
        """Get all document ids from the database.

        Used only for testing purposes.
        """
        docs = self.client.search(
            index=self.index_name, body={"query": {"match_all": {}}, "size": MAX_SIZE}, params={"_source_excludes": ["vector_field"]}
        )
        return [doc["_id"] for doc in docs["hits"]["hits"]]

    def delete_all(self) -> None:
        """Delete all documents from the database.

        Used only for testing purposes.
        """
        if ids := self.get_all_ids():
            self.delete(ids)

    def delete(self, ids: list[str] | None = None, refresh_indices: bool | None = None, **kwargs: Any) -> bool | None:
        """Delete documents from the database.

        We need to reimplement this method since the serverless OpenSearch does not support refresh indices.
        """
        if not ids:
            return None

        refresh_indices = refresh_indices or self.service_name == "es"
        return super().delete(ids, refresh_indices, **kwargs)

    def search_by_vector(self, vector: list[float], k: int = MAX_SIZE, filter_: str | None = None) -> list[Document]:  # type: ignore
        return self.similarity_search_by_vector(embedding=vector, k=k, efficient_filter=filter_ or {}, score_threshold=-1.0)

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Range

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    from ..models.qdrant_input_model import QdrantIntegration


class QdrantDatabase(Qdrant, VectorDbBase):
    def __init__(self, actor_input: QdrantIntegration, embeddings: Embeddings) -> None:
        if actor_input.qdrantAutoCreateCollection:
            # The collection is created if it doesn't exist
            # The text passed is used to determine the dimension of the vector
            # This method is usually called internally by Qdrant#from_documents and Qdrant#from_texts
            Qdrant.construct_instance(
                ["<dummy-text>"],
                embedding=embeddings,
                url=actor_input.qdrantUrl,
                api_key=actor_input.qdrantApiKey,
                collection_name=actor_input.qdrantCollectionName,
            )

        client = QdrantClient(url=actor_input.qdrantUrl, api_key=actor_input.qdrantApiKey)
        super().__init__(client=client, collection_name=actor_input.qdrantCollectionName, embeddings=embeddings)

        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        # noinspection PyBroadException
        try:
            self.client.get_collections()
        except Exception:
            return False
        else:
            return True

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())

        # Qdrant-Langchain nests metadata and document content in the payload.
        # We specify the metadata key to update the nested "last_seen_at" field.
        # https://qdrant.tech/documentation/concepts/payload/#set-payload
        self.client.set_payload(self.collection_name, {"last_seen_at": last_seen_at}, points=ids, key=self.metadata_payload_key)

    def delete_expired(self, expired_ts: int) -> None:
        self.client.delete(
            self.collection_name,
            Filter(
                must=[
                    FieldCondition(
                        key=f"{self.metadata_payload_key}.last_seen_at",
                        range=Range(
                            lt=expired_ts,
                        ),
                    )
                ]
            ),
        )

    def delete_all(self) -> None:
        self.client.delete(self.collection_name, Filter(must=[]))

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: dict | None = None) -> list[Document]:
        return self.similarity_search_by_vector(embedding=vector, k=k, filter=filter_)

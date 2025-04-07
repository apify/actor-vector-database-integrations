from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import delete, text, update
from sqlalchemy.sql.expression import literal

from .base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from ..models import PgvectorIntegration


class PGVectorDatabase(PGVector, VectorDbBase):
    def __init__(self, actor_input: PgvectorIntegration, embeddings: Embeddings) -> None:
        super().__init__(
            embeddings=embeddings, collection_name=actor_input.postgresCollectionName, connection=actor_input.postgresSqlConnectionStr, use_jsonb=True
        )
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
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            store = self.EmbeddingStore
            result = session.query(store).where(store.collection_id == collection.uuid).where(store.id == id_).first()
            return (
                Document(page_content=result.document, metadata=result.cmetadata | {"chunk_id": result.id})
                if result
                else Document(page_content="", metadata={})
            )

    def get_all_ids(self) -> list[str]:
        """Get all document ids from the database.

        Used only for testing purposes.
        """

        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            ids = session.query(self.EmbeddingStore.id).filter(self.EmbeddingStore.collection_id == collection.uuid).all()
            return [r[0] for r in ids]

    def get_by_item_id(self, item_id: str) -> list[Document]:
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            results = (
                session.query(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == collection.uuid)
                .where(text("(cmetadata ->> 'item_id') = :value").bindparams(value=item_id))
                .all()
            )

        return [Document(page_content="", metadata=r.cmetadata | {"chunk_id": r.id}) for r in results]

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())

        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            stmt = (
                update(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == literal(str(collection.uuid)))
                .where(self.EmbeddingStore.id.in_(ids))
                .values(cmetadata=text(f"cmetadata || jsonb_build_object('last_seen_at', {last_seen_at})"))
            )
            session.execute(stmt)
            session.commit()

    def delete_by_item_id(self, item_id: str) -> None:
        """Delete object by item_id."""
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

        stmt = (
            delete(self.EmbeddingStore)
            .where(self.EmbeddingStore.collection_id == literal(str(collection.uuid)))
            .where(text("(cmetadata ->> 'item_id') = :value").bindparams(value=item_id))
        )
        session.execute(stmt)
        session.commit()

    def delete_expired(self, expired_ts: int) -> None:
        """Delete objects from the index that are expired."""

        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            stmt = (
                delete(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == literal(str(collection.uuid)))
                .where(text("(cmetadata ->> 'last_seen_at')::int < :value").bindparams(value=expired_ts))
            )
            session.execute(stmt)
            session.commit()

    def delete_all(self) -> None:
        """Delete all documents from the database.

        Used only for testing purposes.
        """
        if ids := self.get_all_ids():
            self.delete(ids=ids, collection_only=True)

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: dict | None = None) -> list[Document]:
        """Search by vector and return the results."""
        return self.similarity_search_by_vector(vector, k=k, filter=filter_)

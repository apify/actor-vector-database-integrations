from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

BACKOFF_MAX_TIME_SECONDS = 900
BACKOFF_MAX_TIME_DELETE_SECONDS = 900  # 15 minutes (if many objects were added it takes time to search in the database)


class VectorDbBase(ABC):
    # only for testing purposes (to wait for the index to be updated, e.g. in Pinecone)
    unit_test_wait_for_index = 0

    @abstractmethod
    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get documents by item_id."""

    @abstractmethod
    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

    @abstractmethod
    def delete_by_item_id(self, item_id: str) -> None:
        """Delete documents by item_id."""

    @abstractmethod
    def delete_expired(self, expired_ts: int) -> None:
        """Delete documents that are older than the ts_expired timestamp."""

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all documents from the database (internal function for testing purposes)."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the database is connected."""

    @abstractmethod
    def search_by_vector(self, vector: list[float], k: int, filter_: dict | None = None) -> list[Document]:
        """Search for documents by vector. Return a list of documents."""

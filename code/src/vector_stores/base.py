from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


class FailedToConnectToDatabaseError(Exception):
    """Failed to connect to a vector database."""


class VectorDbBase(ABC):
    @abstractmethod
    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database."""

    @abstractmethod
    def delete_expired(self, ts_expired: int) -> None:
        """Delete documents that are older than the ts_expired timestamp."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the database is connected."""

    @abstractmethod
    def search_by_vector(self, vector: list[float], k: int, filter_: dict | None = None) -> list[Document]:
        """Search for documents by vector. Return a list of documents."""

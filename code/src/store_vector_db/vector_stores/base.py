from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


class FailedToConnectToDatabaseError(Exception):
    """Failed to connect to a vector database."""


class VectorDatabaseHelperClass(ABC):

    @abstractmethod
    def update_metadata(self, data: list[Document]) -> None:
        """Update metadata of the documents in the database. Used to update the updated_at field."""

    @abstractmethod
    def delete_orphaned(self, timestamp: int) -> None:
        """Delete documents that are older than the timestamp."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the database is connected."""

    @abstractmethod
    def search_by_vector(self, vector: list[float], k: int, filter_: dict | None = None) -> list[Document]:
        """Search for documents by vector. Return a list of documents."""

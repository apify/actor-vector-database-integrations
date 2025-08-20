# code/tests/test_chroma_batch.py
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .conftest import DATABASE_FIXTURES

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from langchain_core.documents import Document

    from src._types import VectorDb

from langchain_core.documents import Document

from src.vector_stores.chroma import BATCH_SIZE, batch


def _make_docs(n: int) -> list[Document]:
    return [Document(page_content=f"batch {i}", metadata={"chunk_id": f"batch-{i}"}) for i in range(n)]


def test_batch_respects_batch_size() -> None:
    total = BATCH_SIZE * 2 + 5  # two full batches + remainder
    docs = _make_docs(total)

    chunks = list(batch(docs, BATCH_SIZE))

    assert len(chunks) == 3, "Expected 3 batches"
    assert len(chunks[-1]) == 5, "Remainder batch size incorrect"


@pytest.mark.parametrize("bad_size", [0, -1])
def test_batch_invalid_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="size must be > 0"):
        list(batch(_make_docs(1), bad_size))


@pytest.mark.integration()
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
@pytest.mark.skipif("db_chroma" not in DATABASE_FIXTURES, reason="chroma database is not enabled")
def test_add_documents_batches(input_db: str, request: FixtureRequest) -> None:
    # Force small batch size to minimize embeddings/API calls while ensuring multiple batches.

    db: VectorDb = request.getfixturevalue(input_db)
    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 initial objects in the database"

    db.delete_all()  # Clear the database before testing
    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 0, "Expected 0 objects in the database after delete_all"

    total_new = 11  # Will require 3 batches (5 + 5 + 1)
    docs = _make_docs(total_new)

    ids = [doc.metadata["chunk_id"] for doc in docs]
    inserted_ids = db.add_documents(docs, batch_size=5, ids=ids)

    assert len(inserted_ids) == len(ids), "Expected all new documents inserted"
    assert inserted_ids == [d.metadata["chunk_id"] for d in docs], "Order of returned IDs must match input order"

    # Verify they are really stored
    res = db.search_by_vector(db.dummy_vector, k=20)
    print(f"Total objects in the database after batch insert: {len(res)}")
    print(res)
    assert len(res) == 11, "Expected 11 objects in the database after batch insert"

    for doc in res:
        assert isinstance(doc, Document), "Expected each result to be a Document instance"
        assert doc.id in ids, f"Missing inserted id: {doc.id}"

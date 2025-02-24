from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from src.vcs import compare_crawled_data_with_db, delete_expired_objects, update_db_with_crawled_data

from .conftest import DATABASE_FIXTURES, ID1, ID3, ID4A, ID4B, ID4C, ID5A, ID5B, ID5C, ID6, ITEM_ID1, ITEM_ID4

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from langchain_core.documents import Document

    from src._types import VectorDb


def wait_for_db(sec: int = 3) -> None:
    # Wait for the database to update (Pinecone)
    # Data freshness - Pinecone is eventually consistent, so there can be a slight delay before new or changed records are visible to queries.
    time.sleep(sec)


# Helper to compute the expected stored chunk_id
def get_expected_id(db: VectorDb, item_id: str, chunk_id: str) -> str:
    if hasattr(db, "use_id_prefix") and getattr(db, "use_id_prefix", True):
        return f"{item_id}#{chunk_id}"
    return chunk_id


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_add_newly_crawled_data(input_db: str, crawl_2: list[Document], request: FixtureRequest) -> None:
    db: VectorDb = request.getfixturevalue(input_db)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 initial objects in the database"

    data_add, _, _ = compare_crawled_data_with_db(db, crawl_2)

    assert len(data_add) == 4, "Expected 4 objects to add"
    assert data_add[0].metadata["chunk_id"] == ID4C
    assert data_add[1].metadata["chunk_id"] == ID5B
    assert data_add[2].metadata["chunk_id"] == ID5C
    assert data_add[3].metadata["chunk_id"] == ID6

    # Add new data
    db.add_documents(data_add, ids=[d.metadata["chunk_id"] for d in data_add])
    wait_for_db(db.unit_test_wait_for_index)

    id4c = get_expected_id(db, "id4", ID4C)
    id5b = get_expected_id(db, "id5", ID5B)
    id5c = get_expected_id(db, "id5", ID5C)

    res = db.search_by_vector(db.dummy_vector, k=10)
    ids = [r.metadata["chunk_id"] for r in res]
    assert len(res) == 10, "Expected 10 objects in the database after addition"
    assert id4c in ids, f"Expected {id4c} to be added"
    assert id5b in ids, f"Expected {id5b} to be added"
    assert id5c in ids, f"Expected {id5c} to be added"


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_get_by_item_id(input_db: str, request: FixtureRequest) -> None:
    db: VectorDb = request.getfixturevalue(input_db)

    res = db.get_by_item_id(ITEM_ID1)

    id1 = get_expected_id(db, "id1", ID1)
    id4a = get_expected_id(db, "id4", ID4A)
    id4b = get_expected_id(db, "id4", ID4B)

    assert len(res) == 1, "Expected 1 object to be returned"
    assert res[0].metadata["item_id"] == ITEM_ID1, f"Expected {ITEM_ID1} to be returned"
    assert res[0].metadata["chunk_id"] == id1, f"Expected {id1} to be returned"

    res = db.get_by_item_id(ITEM_ID4)

    assert len(res) == 2, "Expected 2 objects to be returned"

    ids = [r.metadata["chunk_id"] for r in res]
    assert id4a in ids, f"Expected {id4a} to be returned"
    assert id4b in ids, f"Expected {id4b} to be returned"

    res = db.get_by_item_id("idX")
    assert not res, "Expected [] to be returned"

    res = db.get_by_item_id("")
    assert not res, "Expected [] to be returned"


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_update_metadata_last_seen_at(input_db: str, crawl_2: list[Document], request: FixtureRequest) -> None:
    # to test whether the object was updated
    ts_init = int(datetime.now(timezone.utc).timestamp())

    db: VectorDb = request.getfixturevalue(input_db)
    id3 = get_expected_id(db, "id3", ID3)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 initial objects in the database"

    _, ids_update_last_seen, _ = compare_crawled_data_with_db(db, crawl_2)

    assert len(ids_update_last_seen) == 1, "Expected 1 object to update"

    # OpenSearch serverless does not support to create a document with ID
    # Therefore, we cannot check the ID directly and need to get the ID from the database
    ids_orig = ids_update_last_seen
    if hasattr(db, "get_by_id") and (v := db.get_by_id(ids_update_last_seen[0])):
        ids_orig = v.metadata["chunk_id"]
    assert id3 in ids_orig, f"Expected {id3} to be updated"

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert next(r for r in res if r.metadata["chunk_id"] == id3).metadata["last_seen_at"] == 1

    # Update metadata data
    db.update_last_seen_at(ids_update_last_seen)
    wait_for_db(db.unit_test_wait_for_index)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 objects in the database after last_seen update"
    assert next(r for r in res if r.metadata["chunk_id"] == id3).metadata["last_seen_at"] >= ts_init, f"Expected {id3} to be updated"


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_delete_updated_data(input_db: str, crawl_2: list[Document], request: FixtureRequest) -> None:
    db: VectorDb = request.getfixturevalue(input_db)

    id4a = get_expected_id(db, "id4", ID4A)
    id4b = get_expected_id(db, "id4", ID4B)
    id5a = get_expected_id(db, "id5", ID5A)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 initial objects in the database"

    _, _, ids_del = compare_crawled_data_with_db(db, crawl_2)

    # OpenSearch serverless does not support to create a document with ID, so we cannot check the ID directly
    # Therefore, we need to get the ID from the database

    ids_orig = ids_del
    if hasattr(db, "get_by_id") and (data := [db.get_by_id(id_) for id_ in ids_del]):
        ids_orig = [d.metadata["chunk_id"] for d in data if d]

    assert len(ids_del) == 3, "Expected 1 object to delete"
    assert id4a in ids_orig, f"Expected {id4a} to be deleted"
    assert id4b in ids_orig, f"Expected {id4b} to be deleted"
    assert id5a in ids_orig, f"Expected {id5a} to be deleted"

    db.delete(ids=ids_del)
    wait_for_db(db.unit_test_wait_for_index)

    res = db.search_by_vector(db.dummy_vector, k=10)
    ids = [r.metadata["chunk_id"] for r in res]
    assert len(ids) == 3, "Expected 3 objects in the database after deletion"
    assert id4a not in ids, f"Expected {id4a} to be deleted"
    assert id4b not in ids, f"Expected {id4b} to be deleted"
    assert id5a not in ids, f"Expected {id5a} to be deleted"


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_deleted_expired_data(input_db: str, request: FixtureRequest) -> None:
    db: VectorDb = request.getfixturevalue(input_db)

    id1 = get_expected_id(db, ITEM_ID1, ID1)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 initial objects in the database"

    # Delete expired objects
    db.delete_expired(expired_ts=1)
    wait_for_db(db.unit_test_wait_for_index)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 5, "Expected 5 objects in the database after deletion"
    assert id1 not in [r.metadata["chunk_id"] for r in res], f"Expected {id1} to be deleted"


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_update_db_with_crawled_data_all(input_db: str, crawl_2: list[Document], expected_results: list[Document], request: FixtureRequest) -> None:
    db: VectorDb = request.getfixturevalue(input_db)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 initial objects in the database"

    update_db_with_crawled_data(db, crawl_2)
    wait_for_db(db.unit_test_wait_for_index)
    delete_expired_objects(db, 1)
    wait_for_db(db.unit_test_wait_for_index)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 6, "Expected 6 objects in the database after all updates"

    # Compare results with expected results
    for expected in expected_results:
        _id = get_expected_id(db, expected.metadata["item_id"], expected.metadata["chunk_id"])
        d = next(r for r in res if _id == r.metadata["chunk_id"])
        assert d.metadata["item_id"] == expected.metadata["item_id"], f"Expected item_id {expected.metadata['item_id']}"
        assert d.metadata["checksum"] == expected.metadata["checksum"], f"Expected checksum {expected.metadata['checksum']}"


@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_get_delete_all(input_db: str, request: FixtureRequest) -> None:
    """Test that all items have benn deleted (delete_all is an internal function)."""

    db: VectorDb = request.getfixturevalue(input_db)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert res

    db.delete_all()
    wait_for_db(db.unit_test_wait_for_index)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert not res

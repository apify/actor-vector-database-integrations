import pytest
import time

from src.constants import VCR_HEADERS_EXCLUDE
from src.vcs import compare_crawled_data_with_db, update_db_with_crawled_data


# Database fixtures to test. Fill here the name of the fixtures you want to test
DATABASE_FIXTURES = ["db_pinecone", "db_chroma"]


def wait_for_db(sec=3):
    # Wait for the database to update (Pinecone)
    # Data freshness - Pinecone is eventually consistent, so there can be a slight delay before new or changed records are visible to queries.
    time.sleep(sec)


@pytest.mark.integration
@pytest.mark.vcr(filter_headers=VCR_HEADERS_EXCLUDE)
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_update_db_with_crawled_data(input_db, crawl_2, request):

    db = request.getfixturevalue(input_db)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 5, "Expected 5 objects in the database"

    data_add, ids_update_last_seen, ids_del = compare_crawled_data_with_db(db, crawl_2)

    assert len(data_add) == 2, "Expected 2 objects to add"
    assert data_add[0].metadata["id"] == "id4#4c"
    assert data_add[1].metadata["id"] == "id5#5"

    assert len(ids_update_last_seen) == 1, "Expected 1 object to update"
    assert "id3#3" in ids_update_last_seen, "Expected id3#3 to be updated"

    assert len(ids_del) == 2, "Expected 1 object to delete"
    assert "id4#4a" in ids_del, "Expected id4#4a to be deleted"
    assert "id4#4b" in ids_del, "Expected id4#4b to be deleted"


@pytest.mark.integration
@pytest.mark.vcr(filter_headers=VCR_HEADERS_EXCLUDE)
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_delete_updated_data(input_db, crawl_2, request):

    db = request.getfixturevalue(input_db)
    _, _, ids_del = compare_crawled_data_with_db(db, crawl_2)

    db.delete(ids=ids_del)
    wait_for_db()
    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 3, "Expected 3 objects in the database after deletion"
    assert "id4#4a" not in [r.metadata["id"] for r in res], "Expected id4#4a to be deleted"
    assert "id4#4b" not in [r.metadata["id"] for r in res], "Expected id4#4b to be deleted"


@pytest.mark.integration
@pytest.mark.vcr(filter_headers=VCR_HEADERS_EXCLUDE)
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_add_newly_crawled_data(input_db, crawl_2, request):

    db = request.getfixturevalue(input_db)
    data_add, _, _ = compare_crawled_data_with_db(db, crawl_2)

    # Add new data
    db.add_documents(data_add, ids=[d.metadata["id"] for d in data_add])
    wait_for_db()
    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 7, "Expected 7 objects in the database after addition"
    assert "id4#4c" in [r.metadata["id"] for r in res], "Expected id4#4c to be added"
    assert "id5#5" in [r.metadata["id"] for r in res], "Expected id5#5 to be added"


@pytest.mark.integration
@pytest.mark.vcr(filter_headers=VCR_HEADERS_EXCLUDE)
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_update_metadata_last_seen_at(input_db, crawl_2, request):

    db = request.getfixturevalue(input_db)
    _, ids_update_last_seen, _ = compare_crawled_data_with_db(db, crawl_2)
    assert "id3#3" in ids_update_last_seen

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert [r for r in res if r.metadata["id"] == "id3#3"][0].metadata["last_seen_at"] == 1

    # Update metadata data
    db.update_last_seen_at(ids_update_last_seen)
    wait_for_db()
    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 5, "Expected 5 objects in the database after update"
    assert [r for r in res if r.metadata["id"] == "id3#3"][0].metadata["last_seen_at"] > 1, "Expected id3#3 to be updated"


@pytest.mark.integration
@pytest.mark.vcr(filter_headers=VCR_HEADERS_EXCLUDE)
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_deleted_expired_data(input_db, crawl_2, request):

    db = request.getfixturevalue(input_db)
    # Delete expired objects
    res = db.search_by_vector(db.dummy_vector, k=10, filter_={"last_seen_at": {"$lt": 1}})
    assert res, "Expected expired objects in the database"
    assert len(res) == 1, "Expected 1 expired object in the database"

    db.delete_expired(ts_expired=1)
    wait_for_db()

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 4, "Expected 4 objects in the database after deletion"
    assert "id1#1" not in [r.metadata["id"] for r in res], "Expected id1#1 to be deleted"


@pytest.mark.integration
@pytest.mark.vcr(filter_headers=VCR_HEADERS_EXCLUDE)
@pytest.mark.parametrize("input_db", DATABASE_FIXTURES)
def test_update_db_with_crawled_data_all(input_db, crawl_2, expected_results, request):

    db = request.getfixturevalue(input_db)
    update_db_with_crawled_data(db, crawl_2, 1)
    wait_for_db(5)

    res = db.search_by_vector(db.dummy_vector, k=10)
    assert len(res) == 4, "Expected 4 objects in the database after all updates"

    # Compare results with expected results
    for expected in expected_results:
        d = [r for r in res if expected.metadata["id"] == r.metadata["id"]][0]
        assert d.metadata["item_id"] == expected.metadata["item_id"], f"Expected item_id {expected.metadata['item_id']}"
        assert d.metadata["checksum"] == expected.metadata["checksum"], f"Expected checksum {expected.metadata['checksum']}"

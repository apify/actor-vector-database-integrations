from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document

from src.models import PineconeIntegration
from src.vcs import update_db_with_crawled_data
from src.vector_stores.pinecone import PineconeDatabase

from .conftest import DATABASE_FIXTURES, INDEX_NAME, embeddings

ID1, ID2 = "1", "2"
ITEM_ID1, ITEM_ID2 = "id1", "id2"

# IDs for the dataset-id-field test (distinct from namespace test IDs above)
DID1, DID2, DID3 = "did1", "did2", "did3"
DITEM_ID1, DITEM_ID2 = "ditem1", "ditem2"

# Documents that include a dataset "id" field in metadata, reproducing the bug where
# datasetFields contains "id" (e.g. ['id', 'url', 'page_content']).
# The "id" field must NOT be used as the vector store key — chunk_id must be used instead.
dd1 = Document(
    page_content="doc1_v1", metadata={"id": "dataset-hash-1", "item_id": DITEM_ID1, "chunk_id": DID1, "checksum": "cs1", "last_seen_at": 1}
)
dd2 = Document(page_content="doc2", metadata={"id": "dataset-hash-2", "item_id": DITEM_ID2, "chunk_id": DID2, "checksum": "cs2", "last_seen_at": 1})
# Updated version of dd1: same item_id, different checksum → old chunk deleted, new chunk added
dd1_updated = Document(
    page_content="doc1_v2", metadata={"id": "dataset-hash-1", "item_id": DITEM_ID1, "chunk_id": DID3, "checksum": "cs1-new", "last_seen_at": 2}
)

NAMESPACE1 = "namespace1"
NAMESPACE2 = "namespace2"

d1 = Document(page_content="namespace_1", metadata={"item_id": ITEM_ID1, "chunk_id": ID1, "checksum": "1", "last_seen_at": 0})
d2 = Document(page_content="namespace_2", metadata={"item_id": ITEM_ID2, "chunk_id": ID2, "checksum": "2", "last_seen_at": 1})


def wait_for_db(sec: int = 3) -> None:
    # Wait for the database to update (Pinecone)
    # Data freshness - Pinecone is eventually consistent, so there can be a slight delay before new or changed records are visible to queries.
    time.sleep(sec)


@pytest.mark.integration()
@pytest.mark.skipif("db_pinecone" not in DATABASE_FIXTURES, reason="pinecone database is not enabled")
@pytest.fixture()
def db_pinecone_ns() -> PineconeDatabase:  # type: ignore
    db = PineconeDatabase(
        actor_input=PineconeIntegration(  # type: ignore
            pineconeIndexName=INDEX_NAME,
            pineconeIndexNamespace=NAMESPACE1,
            pineconeApiKey=os.getenv("PINECONE_API_KEY") or "fake",
            embeddingsProvider="OpenAI",
            embeddingsApiKey=os.getenv("OPENAI_API_KEY") or "fake",
            datasetFields=["text"],
        ),
        embeddings=embeddings,
    )

    def delete_ns(namespace: str) -> None:
        if r := list(db.index.list(prefix="", namespace=namespace)):
            db.delete(ids=r, namespace=namespace)

    db.unit_test_wait_for_index = 10

    # delete_all is not deleting all namespaces
    db.delete_all()
    delete_ns("default")
    delete_ns(NAMESPACE1)
    delete_ns(NAMESPACE2)
    wait_for_db(db.unit_test_wait_for_index)

    yield db

    db.delete_all()
    delete_ns("default")
    delete_ns(NAMESPACE1)
    delete_ns(NAMESPACE2)


@pytest.mark.integration()
@pytest.mark.skipif("db_pinecone" not in DATABASE_FIXTURES, reason="pinecone database is not enabled")
def test_namespace(db_pinecone_ns: PineconeDatabase) -> None:
    """Test namespace functionality

    Add 1 document to the default namespace (created with database constructor) and 2 documents to another namespace.
    Perform search, get, delete operations on the default namespace and check the other namespace is not affected.
    """

    db: PineconeDatabase = db_pinecone_ns
    ns2 = NAMESPACE2

    r1 = db.search_by_vector(db.dummy_vector, k=10)
    assert len(r1) == 0, "Expected 0 initial objects in the database"

    db.add_documents(documents=[d1], ids=[ID1])
    wait_for_db(db.unit_test_wait_for_index)

    r2 = db.search_by_vector(db.dummy_vector, k=10)
    assert len(r2) == 1, f"Expected 1 objects in the database namespace: {db.namespace}"

    r3 = db.similarity_search_by_vector_with_score(db.dummy_vector, namespace=ns2)
    assert len(r3) == 0, f"Expected 0 objects in the database namespace: {ns2}"

    db.add_documents(documents=[d1, d2], ids=[ID1, ID2], namespace=ns2)
    wait_for_db(db.unit_test_wait_for_index)

    r4 = db.search_by_vector(db.dummy_vector, k=10)
    assert len(r4) == 1, f"Expected 1 objects in the database namespace: {db.namespace}"

    r5 = db.similarity_search_by_vector_with_score(db.dummy_vector, namespace=ns2)
    assert len(r5) == 2, f"Expected 2 objects in the database namespace: {ns2}"

    # Get item by id
    assert len(db.get_by_item_id(ITEM_ID1)) == 1, f"Expected 1 object to be returned from {db.namespace}"
    assert len(db.get_by_item_id(ITEM_ID2)) == 0, f"Expected 0 object to be returned from {db.namespace}"

    # Update last_seen_at for namespace1 (check that namespace2 is not affected)
    db.update_last_seen_at([ID1])
    wait_for_db(db.unit_test_wait_for_index)

    r6 = db.search_by_vector(db.dummy_vector, k=10)
    assert next(r for r in r6 if r.metadata["chunk_id"] == ID1).metadata["last_seen_at"] != 1, f"Expected {ID1} to be updated"

    r7 = db.similarity_search_by_vector_with_score(db.dummy_vector, namespace=ns2)
    assert next(r for r, _ in r7 if r.metadata["chunk_id"] == ID1).metadata["last_seen_at"] == 0, f"Expected {ID1} to be untouched"
    assert next(r for r, _ in r7 if r.metadata["chunk_id"] == ID2).metadata["last_seen_at"] == 1, f"Expected {ID2} to be untouched"

    # Delete expired item from namespace1 (check that namespace2 is not affected)
    db.delete_expired(int(datetime.now(timezone.utc).timestamp()) + 1)
    wait_for_db(db.unit_test_wait_for_index)

    r8 = db.search_by_vector(db.dummy_vector, k=10)
    assert len(r8) == 0, f"Expected 0 objects after delete in the database namespace: {db.namespace}"

    r9 = db.similarity_search_by_vector_with_score(db.dummy_vector, namespace=ns2)
    assert len(r9) == 2, f"Expected 2 objects after delete in the database namespace: {ns2}"


@pytest.fixture()
def db_pinecone_dataset_id_field() -> PineconeDatabase:  # type: ignore
    db = PineconeDatabase(
        actor_input=PineconeIntegration(  # type: ignore
            pineconeIndexName=INDEX_NAME,
            pineconeApiKey=os.getenv("PINECONE_API_KEY") or "fake",
            embeddingsProvider="OpenAI",
            embeddingsApiKey=os.getenv("OPENAI_API_KEY") or "fake",
            datasetFields=["id", "text"],
        ),
        embeddings=embeddings,
    )
    db.unit_test_wait_for_index = 20

    db.delete_all()
    db.add_documents(documents=[dd1, dd2], ids=[dd1.metadata["chunk_id"], dd2.metadata["chunk_id"]])
    wait_for_db(db.unit_test_wait_for_index)

    yield db

    db.delete_all()
    wait_for_db(db.unit_test_wait_for_index)


@pytest.mark.integration()
@pytest.mark.skipif("db_pinecone" not in DATABASE_FIXTURES, reason="pinecone database is not enabled")
def test_delta_update_with_dataset_id_field(db_pinecone_dataset_id_field: PineconeDatabase) -> None:
    """Regression test: delta updates must use chunk_id (the vector store key), not the dataset "id" field.

    When a dataset item has "id" in its fields (e.g. datasetFields=['id', 'url', 'page_content']),
    the value gets stored in vector metadata. The old code keyed on metadata["id"] first, which
    picked up the dataset value instead of the vector store ID, causing deletes and last_seen_at
    updates to silently do nothing. End result: stale records were never removed, inflating the DB.
    """
    db: PineconeDatabase = db_pinecone_dataset_id_field

    r1 = db.search_by_vector(db.dummy_vector, k=10)
    assert len(r1) == 2, "Expected 2 initial objects in the database"

    # dd2 is unchanged (same checksum), dd1 has a new version (different checksum)
    update_db_with_crawled_data(db, [dd1_updated, dd2])
    wait_for_db(db.unit_test_wait_for_index)

    r2 = db.search_by_vector(db.dummy_vector, k=10)
    chunk_ids = {r.metadata["chunk_id"] for r in r2}

    assert len(r2) == 2, (
        f"Expected 2 objects after delta update (old dd1 deleted, new dd1 added, dd2 unchanged). "
        f"Got {len(r2)} — stale records were likely not deleted because metadata['id'] was used instead of chunk_id."
    )
    assert DID1 not in chunk_ids, f"Old chunk {DID1} should have been deleted"
    assert DID3 in chunk_ids, f"New chunk {DID3} should have been added"
    assert DID2 in chunk_ids, f"Unchanged chunk {DID2} should still be present"

# type: ignore
"""
This script serves as a playground for playing with ChromaDB.

It demonstrates the process of performing delta updates on ChromaDB. The process is as follows:
1. The database is initially populated with a set of crawled data (`crawl_1`).
2. A new set of data, `crawl_2`, is then crawled and compared with the existing data in the database.
3. The script contains several checks to validate that the database is updated correctly based on the comparison between `crawl_1` and `crawl_2`.

Run as a module:
    python -m src.examples.2024-05-30-chromadb
"""
import asyncio
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from .data_examples_uuid import ID1, ID3, ID4A, ID4B, ID4C, ID5A, ID5B, ID5C, ID6, crawl_1, crawl_2, expected_results
from ..vcs import compare_crawled_data_with_db
from ..models.chroma_input_model import ChromaIntegration
from ..vector_stores.chroma import ChromaDatabase

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True

db = ChromaDatabase(
    ChromaIntegration(
        chromaCollectionName=os.getenv("CHROMA_COLLECTION_NAME"),
        chromaClientHost=os.getenv("CHROMA_CLIENT_HOST"),
        chromaClientPort=int(os.getenv("CHROMA_CLIENT_PORT", "8000")),
        chromaClientSsl=os.getenv("CHROMA_CLIENT_SSL", "false").lower() == "true",
        chromaApiToken=os.getenv("CHROMA_API_TOKEN"),
        chromaTenant=os.getenv("CHROMA_TENANT"),
        chromaDatabase=os.getenv("CHROMA_DATABASE"),
        embeddingsProvider="OpenAI",
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)
index = db.index

print("Database is connected: ", asyncio.run(db.is_connected()))


def wait_for_index(sec=1):
    time.sleep(sec)


if DROP_AND_INSERT:
    db.delete_all()
    r = db.similarity_search("text", k=100)
    print("Initial results count:", len(r))

    inserted = db.add_documents(documents=crawl_1, ids=[d.metadata["chunk_id"] for d in crawl_1])
    print("Inserted ids:", inserted)

r = db.similarity_search("text", k=100)
print("Search results:", r)
print("Search results count:", len(r))


res = db.search_by_vector(db.dummy_vector, k=10)
print("Objects in the database:", len(res), res)
assert len(res) == 6, "Expected 6 objects in the database"

data_add, ids_update_last_seen, ids_del = compare_crawled_data_with_db(db, crawl_2)

print("Data to add", data_add)
print("Ids to update", ids_update_last_seen)
print("Ids to delete", ids_del)

# Update assertions based on actual test data
assert len(data_add) == 4, "Expected 4 objects to add"
# Verify the specific IDs that should be added
expected_add_ids = ['00000000-0000-0000-0000-00000000004c', '00000000-0000-0000-0000-00000000005b', '00000000-0000-0000-0000-00000000005c', '00000000-0000-0000-0000-000000000060']
for doc in data_add:
    assert doc.metadata["chunk_id"] in expected_add_ids, f"Unexpected document to add: {doc.metadata['chunk_id']}"

assert len(ids_update_last_seen) == 1, "Expected 1 object to update"
assert '00000000-0000-0000-0000-000000000030' in ids_update_last_seen, f"Expected {'00000000-0000-0000-0000-000000000030'} to be updated"

assert len(ids_del) == 3, "Expected 3 objects to delete"
expected_del_ids = ['00000000-0000-0000-0000-00000000004a', '00000000-0000-0000-0000-00000000004b', '00000000-0000-0000-0000-00000000005a']
for del_id in ids_del:
    assert del_id in expected_del_ids, f"Unexpected ID to delete: {del_id}"

# Delete data that were updated
db.delete(ids_del)
wait_for_index()
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after delete: ", len(res), res)
assert len(res) == 3, "Expected 3 objects in the database after deletion"

# Add new data
r = db.add_documents(data_add, ids=[d.metadata["chunk_id"] for d in data_add])
wait_for_index()
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after adding new", len(res), res)

ids = [r.metadata["chunk_id"] for r in res]
assert len(res) == 7, "Expected 7 objects in the database after addition"
# Verify the specific IDs that should be present
expected_ids_after_add = ['00000000-0000-0000-0000-000000000020', '00000000-0000-0000-0000-000000000010', '00000000-0000-0000-0000-000000000030', '00000000-0000-0000-0000-00000000004c', '00000000-0000-0000-0000-00000000005b', '00000000-0000-0000-0000-00000000005c', '00000000-0000-0000-0000-000000000060']
for expected_id in expected_ids_after_add:
    assert expected_id in ids, f"Expected {expected_id} to be present after addition"

# Update data
ts = int(datetime.now(timezone.utc).timestamp())
res = db.search_by_vector(db.dummy_vector, k=10)
assert next(r for r in res if r.metadata["chunk_id"] == ID3).metadata["last_seen_at"] == 1

# Update metadata data
db.update_last_seen_at(ids_update_last_seen)
wait_for_index()

res = db.search_by_vector(db.dummy_vector, k=10)
assert len(res) == 7, "Expected 7 objects in the database after last_seen update"
assert next(r for r in res if r.metadata["chunk_id"] == ID3).metadata["last_seen_at"] >= ts, f"Expected {ID3} to be updated"

# delete expired objects
db.delete_expired(expired_ts=1)
wait_for_index()

res = db.search_by_vector(db.dummy_vector, k=10)
res = [r for r in res]
print("Database objects after all updates", len(res), res)
assert len(res) == 6, "Expected 6 objects in the database after all updates"
assert next((r for r in res if r.metadata["chunk_id"] == ID1), None) is None, f"Expected {ID1} to be deleted"

# compare results with expected results
print("Expected results count:", len(expected_results))
for r in expected_results:
    d = db.get(r.metadata["chunk_id"])
    metadata = d.get('metadatas', {})[0]
    assert metadata["item_id"] == r.metadata["item_id"], f"Expected item_id {r.metadata['item_id']}"
    assert metadata["checksum"] == r.metadata["checksum"], f"Expected checksum {r.metadata['checksum']}"

print("DONE")

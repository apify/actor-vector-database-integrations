# type: ignore
"""
This script serves as a playground for playing with Pinecone.

It demonstrates the process of performing delta updates on Pinecone. The process is as follows:
1. The database is initially populated with a set of crawled data (`crawl_1`).
2. A new set of data, `crawl_2`, is then crawled and compared with the existing data in the database.
3. The script contains several checks to validate that the database is updated correctly based on the comparison between `crawl_1` and `crawl_2`.

Run as a module:
    python -m src.examples.2024-05-30-pinecone
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from .data_examples import crawl_1, crawl_2, expected_results
from ..models.pinecone_input_model import EmbeddingsProvider, PineconeIntegration
from ..vcs import compare_crawled_data_with_db
from ..vector_stores.pinecone import PineconeDatabase

load_dotenv()
PINECONE_INDEX_NAME = "apify"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True

db = PineconeDatabase(
    actor_input=PineconeIntegration(
        pineconeIndexName=PINECONE_INDEX_NAME,
        pineconeApiKey=os.getenv("PINECONE_API_KEY"),
        embeddingsProvider=EmbeddingsProvider.OpenAI,
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)


def wait_for_index(sec=3):
    time.sleep(sec)


if DROP_AND_INSERT:
    r = list(db.index.list(prefix=""))
    print("Objects in database", r)
    if r:
        db.delete(ids=r)
        print("Deleted all objects from the database")

    # Insert objects
    inserted = db.add_documents(documents=crawl_1, ids=[d.metadata["id"] for d in crawl_1])
    print("Inserted ids:", inserted)
    print("Waiting for indexing")
    wait_for_index(10)
    print("Database stats", db.index.describe_index_stats())


res = db.search_by_vector(db.dummy_vector, k=10)
print("Objects in the database:", len(res), res)
assert len(res) == 5, "Expected 5 objects in the database"

data_add, ids_update_meta, ids_del = compare_crawled_data_with_db(db, crawl_2)

print("Data to add", data_add)
print("Ids to update", ids_update_meta)
print("Ids to delete", ids_del)

assert len(data_add) == 2, "Expected 2 objects to add"
assert data_add[0].metadata["id"] == "id4#4c"
assert data_add[1].metadata["id"] == "id5#5"

assert len(ids_update_meta) == 1
assert "id3#3" in ids_update_meta

assert len(ids_del) == 2
assert "id4#4a" in ids_del
assert "id4#4b" in ids_del

# Delete data that were updated
db.delete(ids_del)
wait_for_index()
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after delete: ", len(res), res)
assert len(res) == 3, "Expected 3 objects in the database after deletion"

# Add new data
r = db.add_documents(data_add, ids=[d.metadata["id"] for d in data_add])
wait_for_index()
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after adding new", len(res), res)
assert len(res) == 5, "Expected 5 objects in the database after addition"

# Update data
db.update_last_seen_at(ids_update_meta)
wait_for_index()
res = db.search_by_vector(db.dummy_vector)
assert [r for r in res if r.metadata["id"] == "id3#3"][0].metadata["last_seen_at"] > 1, "Expected id3#3 to be updated"

res = db.search_by_vector(db.dummy_vector, k=10, filter_={"last_seen_at": {"$lt": 1}})
print("Expired objects in the database", len(res), res)
assert len(res) == 1, "Expected 1 expired object in the database"

# delete expired objects
db.delete_expired(expired_ts=1)
wait_for_index()

res = db.search_by_vector(db.dummy_vector, k=10)
res = [r for r in res]
print("Database objects after all updates", len(res), res)
assert len(res) == 4, "Expected 4 objects in the database after all updates"

# compare results with expected results
for r in expected_results:
    v = db.index.fetch(ids=[r.metadata["id"]])
    d = v["vectors"][r.metadata["id"]]
    assert d.metadata["item_id"] == r.metadata["item_id"], f"Expected item_id {r.metadata['item_id']}"
    assert d.metadata["checksum"] == r.metadata["checksum"], f"Expected checksum {r.metadata['checksum']}"

print("DONE")

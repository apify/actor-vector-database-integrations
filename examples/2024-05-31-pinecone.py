import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore  # type: ignore
from pinecone import Pinecone as PineconeClient  # type: ignore

from data_examples import crawl_1, crawl_2, expected_results
from models.pinecone_input_model import EmbeddingsProvider, PineconeIntegration
from vcs import compare_crawled_data_with_db
from vector_stores.pinecone import PineconeDatabase

load_dotenv(Path.cwd() / ".." / "code" / ".env")
PINECONE_INDEX_NAME = "apify-unit-test"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True

db = PineconeDatabase(
    actor_input=PineconeIntegration(
        pineconeIndexName=PINECONE_INDEX_NAME,
        pineconeApiKey=os.getenv("PINECONE_API_KEY"),
        embeddingsProvider=EmbeddingsProvider.OpenAIEmbeddings,
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)


def wait_for_index(sec=3):
    time.sleep(sec)


if DROP_AND_INSERT:
    r = list(db.index.list(prefix="i"))
    print("Objects in database", r)
    if r:
        db.delete(ids=r)
        print("Deleted all objects from the database")

    # Insert objects
    inserted = db.add_documents(documents=crawl_1, ids=[d.metadata["id"] for d in crawl_1])
    print("Inserted ids:", inserted)
    print("Waiting for indexing")
    wait_for_index(5)
    print("Database stats", db.index.describe_index_stats())


res = db.search_by_vector(db.dummy_vector, k=10)
print("Objects in the database:", len(res), res)
assert len(res) == 4, "Expected 4 objects in the database"

data_add, data_update_last_seen, data_del = compare_crawled_data_with_db(db, crawl_2)

print("Data to add", data_add)
print("Data to update", data_update_last_seen)
print("Data to delete", data_del)

assert len(data_add) == 2, "Expected 2 objects to add"
assert data_add[0].metadata["id"] == "id4#6"
assert data_add[1].metadata["id"] == "id5#5"

assert len(data_update_last_seen) == 1
assert data_update_last_seen[0].metadata["id"] == "id3#3"

assert len(data_del) == 1
assert data_del[0].metadata["id"] == "id4#4"

# Delete data that were updated
db.delete(ids=[d.metadata["id"] for d in data_del])
wait_for_index()
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after delete: ", len(res), res)
assert len(res) == 3, "Expected 3 objects in the database after deletion"

# Add new data
r = db.add_documents(data_add, ids=[d.metadata["id"] for d in data_add])
wait_for_index()
print("Added new crawled and updated objects", len(r), r)
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after adding new", len(res), res)
assert len(res) == 5, "Expected 5 objects in the database after addition"

# Update data
db.update_last_seen_at(data_update_last_seen)
wait_for_index()
# delete expired objects - not supported by serverless index
# r = index.delete(filter={"last_seen_at": {"$lt": 1}})
# print("Deleted expired objects:", r)

res = db.search_by_vector(db.dummy_vector, k=10, filter_={"last_seen_at": {"$lt": 1}})
print("Expired objects in the database", len(res), res)
assert len(res) == 1, "Expected 1 expired object in the database"

# delete expired objects
db.delete(ids=[d.metadata["id"] for d in res])
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
    assert d.metadata["last_seen_at"] == r.metadata["last_seen_at"], f"Expected last_seen_at {r.metadata['last_seen_at']}"

print("All tests passed")
# # Pinecone API - list objects -> return only IDS
# print("List")
# for r in index.list(prefix="item_id", limit=1):
#     print(r)
#
# # Pinecone API - fetch objects -> returns also vectors
# print("Fetch vector")
# print(index.fetch(ids=ids))

# Pinecone API - find objects -> returns only ids, score
# res = index.query(vector=embeddings.embed_query("apify"), top_k=10_000, filter={"item_id": "item_id2"})
# print("Query results", res)
# matches = [match["id"] for match in res["matches"]]
# print("Object ids", matches)

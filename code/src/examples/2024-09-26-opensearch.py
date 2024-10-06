# type: ignore
"""
This script serves as a playground for trying Amazon OpenSearch Serverless or Open Search without AWS.

It demonstrates the process of performing delta updates on Opensearch. The process is as follows:
1. The database is initially populated with a set of crawled data (`crawl_1`).
2. A new set of data, `crawl_2`, is then crawled and compared with the existing data in the database.
3. The script contains several checks to validate that the database is updated correctly based on the comparison between `crawl_1` and `crawl_2`.

Run as a module:
    python -m src.examples.2024-07-08-milvus
"""

import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from models import OpensearchIntegration
from .data_examples_uuid import ID1, ID3, ID4A, ID4B, ID4C, ID5A, ID5B, ID5C, ID6, crawl_1, crawl_2, expected_results
from ..models import EmbeddingsProvider
from ..vcs import compare_crawled_data_with_db
from ..vector_stores.opensearch import OpenSearchDatabase

load_dotenv()
INDEX_NAME = "apify-index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# index_body = {
#     "settings": {"index.knn": True},
#     "mappings": {
#         "properties": {"vector_field": {"type": "knn_vector", "dimension": 1536, "method": {"engine": "faiss", "name": "hnsw", "space_type": "l2"}}}
#     },
# }

useAws4Auth = os.getenv("OPENSEARCH_URL", "").endswith("amazonaws.com")

DROP_AND_INSERT = True

db = OpenSearchDatabase(
    actor_input=OpensearchIntegration(
        useAWS4Auth=useAws4Auth,
        awsAccessKeyId=os.getenv("AWS_ACCESS_KEY_ID"),
        awsSecretAccessKey=os.getenv("AWS_SECRET_ACCESS_KEY"),
        openSearchIndexName=INDEX_NAME,
        openSearchUrl=os.getenv("OPENSEARCH_URL"),
        embeddingsProvider=EmbeddingsProvider.OpenAI.value,
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
        useSsl=useAws4Auth,
        verifyCerts=useAws4Auth,
    ),
    embeddings=embeddings,
)

WAIT_FOR_INDEX = 80 if useAws4Auth else 5

def wait_for_index(sec=WAIT_FOR_INDEX):
    print("Waiting for index to be updated...")
    time.sleep(sec)


if DROP_AND_INSERT:

    db.delete_all()
    wait_for_index()
    r = db.similarity_search("text", k=100)
    print("Initial results count:", len(r))

    db.add_documents(documents=crawl_1, ids=[d.metadata["chunk_id"] for d in crawl_1])
    wait_for_index()

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

assert len(data_add) == 4, "Expected 4 objects to add"
assert data_add[0].metadata["chunk_id"] == ID4C
assert data_add[1].metadata["chunk_id"] == ID5B
assert data_add[2].metadata["chunk_id"] == ID5C
assert data_add[3].metadata["chunk_id"] == ID6

assert len(ids_update_last_seen) == 1, "Expected 1 object to update"
# OpenSearch does not support to create a document with ID, so we cannot check the ID directly
assert ID3 == db.get_by_id(ids_update_last_seen[0]).metadata["chunk_id"]

# OpenSearch does not support to create a document with ID, so we cannot check the ID directly
orig_ids_to_del = [db.get_by_id(id_).metadata["chunk_id"] for id_ in ids_del]
assert len(ids_del) == 3, "Expected 1 object to delete"
assert ID4A in orig_ids_to_del, f"Expected {ID4A} to be deleted"
assert ID4B in orig_ids_to_del, f"Expected {ID4B} to be deleted"
assert ID5A in orig_ids_to_del, f"Expected {ID5A} to be deleted"

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
assert ID4C in ids, f"Expected {ID4C} to be added"
assert ID5B in ids, f"Expected {ID5B} to be added"
assert ID5C in ids, f"Expected {ID5C} to be added"

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
for r in expected_results:
    # We need to search by item_id as chunk_id is UUID and cannot be used for search
    d = [d for d in db.get_by_item_id(r.metadata["item_id"]) if r.metadata["chunk_id"] == d.metadata["chunk_id"]]
    print("Expected", r)
    print(d)
    metadata = d[0].metadata
    print(metadata)
    assert metadata["item_id"] == r.metadata["item_id"], f"Expected item_id {r.metadata['item_id']}"
    assert metadata["checksum"] == r.metadata["checksum"], f"Expected checksum {r.metadata['checksum']}"

print("DONE")

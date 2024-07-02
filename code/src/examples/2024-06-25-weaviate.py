# type: ignore
"""
This script serves as a playground for playing with Weaviate.

It demonstrates the process of performing delta updates on Weaviate. The process is as follows:
1. The database is initially populated with a set of crawled data (`crawl_1`).
2. A new set of data, `crawl_2`, is then crawled and compared with the existing data in the database.
3. The script contains several checks to validate that the database is updated correctly based on the comparison between `crawl_1` and `crawl_2`.

Run as a module:
    python -m src.examples.2024-06-25-weaviate
"""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from .data_examples_uuid import ID1, ID3, ID4A, ID4B, ID4C, ID5A, ID5B, ID5C, ID6, crawl_1, crawl_2, expected_results
from ..models import EmbeddingsProvider, WeaviateIntegration
from ..vcs import compare_crawled_data_with_db
from ..vector_stores.weaviate import WeaviateDatabase

load_dotenv()
WEAVIATE_COLLECTION_NAME = "apify"


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# client = weaviate.connect_to_local()
# Connect to a WCS instance
# client = weaviate.connect_to_wcs(
#     cluster_url=os.getenv("WEAVIATE_URL"),
#     auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
# )
#
# db = WeaviateVectorStore(client=client, index_name=WEAVIATE_COLLECTION_NAME, text_key="text", embedding=embeddings)


DROP_AND_INSERT = True

db = WeaviateDatabase(
    actor_input=WeaviateIntegration(
        weaviateUrl=os.getenv("WEAVIATE_URL"),
        weaviateApiKey=os.getenv("WEAVIATE_API_KEY"),
        weaviateCollectionName=WEAVIATE_COLLECTION_NAME,
        embeddingsProvider=EmbeddingsProvider.OpenAI.value,
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)

if DROP_AND_INSERT:
    db.delete_all()
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

assert len(data_add) == 4, "Expected 4 objects to add"
assert data_add[0].metadata["chunk_id"] == ID4C
assert data_add[1].metadata["chunk_id"] == ID5B
assert data_add[2].metadata["chunk_id"] == ID5C
assert data_add[3].metadata["chunk_id"] == ID6

assert len(ids_update_last_seen) == 1, "Expected 1 object to update"
assert ID3 in ids_update_last_seen, f"Expected {ID3} to be updated"

assert len(ids_del) == 3, "Expected 1 object to delete"
assert ID4A in ids_del, f"Expected {ID4A} to be deleted"
assert ID4B in ids_del, f"Expected {ID4B} to be deleted"
assert ID5A in ids_del, f"Expected {ID5A} to be deleted"

# Delete data that were updated
db.delete(ids_del)
res = db.search_by_vector(db.dummy_vector, k=10)
print("Database objects after delete: ", len(res), res)
assert len(res) == 3, "Expected 3 objects in the database after deletion"

# Add new data
r = db.add_documents(data_add, ids=[d.metadata["chunk_id"] for d in data_add])
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

res = db.search_by_vector(db.dummy_vector, k=10)
assert len(res) == 7, "Expected 7 objects in the database after last_seen update"
assert next(r for r in res if r.metadata["chunk_id"] == ID3).metadata["last_seen_at"] >= ts, f"Expected {ID3} to be updated"

# delete expired objects
db.delete_expired(expired_ts=1)

res = db.search_by_vector(db.dummy_vector, k=10)
res = [r for r in res]
print("Database objects after all updates", len(res), res)
assert len(res) == 6, "Expected 6 objects in the database after all updates"
assert next((r for r in res if r.metadata["chunk_id"] == ID1), None) is None, f"Expected {ID1} to be deleted"

# compare results with expected results
for r in expected_results:
    d = db.get(r.metadata["chunk_id"])
    metadata = d.properties
    assert metadata["item_id"] == r.metadata["item_id"], f"Expected item_id {r.metadata['item_id']}"
    assert metadata["checksum"] == r.metadata["checksum"], f"Expected checksum {r.metadata['checksum']}"

print("DONE")

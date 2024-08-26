# type: ignore
"""
This script serves as a playground for playing with VectorX database.

It demonstrates the process of performing delta updates on Milvus. The process is as follows:
1. The database is initially populated with a set of crawled data (`crawl_1`).
2. A new set of data, `crawl_2`, is then crawled and compared with the existing data in the database.
3. The script contains several checks to validate that the database is updated correctly based on the comparison between `crawl_1` and `crawl_2`.

Notes:
    - VectorX supports vectors upto 2000 dimensions-this is limiting when one wants to use OpenAI embeddings.
    - In the vecx-langchain add correct type - vectorx_index: Optional[Any] = None,
    - In the vecx-langchain db.add_documents - it expects 'source' field in metadata (this is not standard)
    - With wrong dimension  File "/apify/store-vector-db/code/.venv/lib/python3.11/site-packages/vecx_langchain/base.py", line 90, in add_texts
          self._vectorx_index.upsert(entries)
        File "/apify/store-vector-db/code/.venv/lib/python3.11/site-packages/vecx/index.py", line 43, in upsert
           raise_exception(response.status_code)
        File "/apify/store-vector-db/code/.venv/lib/python3.11/site-packages/vecx/exceptions.py", line 82, in raise_exception
            raise APIException(message)
        vecx.exceptions.APIException: API Error: None
    - In the vecx-langchain - prints instead of log.
    - In the vecx-langchain - query_with_filter - limit only 1000 results
    - In the vecx-langchain -  self._vectorx_index.delete(ids=chunk) - vectorx_index has no attribute delete
    - Unable to update metadata using filter

Run as a module:
    python -m src.examples.2024-08-26-vectorx
"""
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from models.vectorx_input_model import VectorxIntegration
from vector_stores.vectorx import VectorxDatabase
from .data_examples_uuid import ID1, ID3, ID4A, ID4B, ID4C, ID5A, ID5B, ID5C, ID6, crawl_1, crawl_2, expected_results
from ..models import EmbeddingsProvider
from ..vcs import compare_crawled_data_with_db
from vecx import vectorx, index

from vecx_langchain import VectorXVectorStore

load_dotenv()
MILVUS_COLLECTION_NAME = "apify"


DIMENSION = 1536
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True


# token = os.getenv("VECTORX_TOKEN")
# vx = vectorx.VectorX(token)

# key = vx.generate_key()
# key = os.getenv("VECTORX_KEY")

INDEX_NAME = "apify"

# CREATE INDEX
# resp = vx.create_index(name="apify", key=key, dimensions=DIMENSION, distance_metric="COSINE", region="india-west")
# print(resp)
# vx.delete_index(key)


db = VectorxDatabase(
    actor_input=VectorxIntegration(
        vectorxToken=os.getenv("VECTORX_TOKEN"),
        vectorxKey=os.getenv("VECTORX_KEY"),
        vectorxIndexName=INDEX_NAME,
        embeddingsProvider=EmbeddingsProvider.OpenAI.value,
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)

# idx = vx.get_index(name=INDEX_NAME, key=key)
# db1 = VectorXVectorStore(idx, embeddings, text_key="text")
# db1.similarity_search()


def wait_for_index(sec=1):
    time.sleep(sec)


if DROP_AND_INSERT:

    print("Deleted all data")
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

exit(0)

# Update data
ts = int(datetime.now(timezone.utc).timestamp())
res = db.search_by_vector(db.dummy_vector, k=10)
assert next(r for r in res if r.metadata["chunk_id"] == ID3).metadata["last_seen_at"] == 1

# Update metadata data
db.update_last_seen_at(ids_update_last_seen)
wait_for_index()

res = db.search_by_vector(db.dummy_vector, k=10)
assert len(res) == 7, "Expected 7 objects in the database after last_seen update"
assert (
    next(r for r in res if r.metadata["chunk_id"] == ID3).metadata["last_seen_at"] >= ts
), f"Expected {ID3} to be updated"

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
    d = db.get(r.metadata["chunk_id"])
    metadata = d[0]
    assert metadata["item_id"] == r.metadata["item_id"], f"Expected item_id {r.metadata['item_id']}"
    assert metadata["checksum"] == r.metadata["checksum"], f"Expected checksum {r.metadata['checksum']}"

print("DONE")

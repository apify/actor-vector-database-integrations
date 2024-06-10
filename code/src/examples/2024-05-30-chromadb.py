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

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from ..models.chroma_input_model import ChromaIntegration
from ..models.pinecone_input_model import EmbeddingsProvider
from ..vcs import compare_crawled_data_with_db, get_vector_store
from .data_examples import crawl_1, crawl_2, expected_results

load_dotenv()
CHROMA_COLLECTION_NAME = "apify"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True

db = asyncio.run(
    get_vector_store(
        ChromaIntegration(
            chromaCollectionName=CHROMA_COLLECTION_NAME,
            chromaClientHost="localhost",
            embeddingsProvider=EmbeddingsProvider.OpenAIEmbeddings.value,
            datasetFields=["text"],
        ),
        embeddings=embeddings,
    )
)
index = db.index

print("Database is connected: ", asyncio.run(db.is_connected()))


if DROP_AND_INSERT:
    r = index.get()
    print("Objects in database", r)
    if r["ids"]:
        index.delete(ids=r["ids"])
        print("Deleted all objects from the database")

    # Insert objects
    inserted = db.add_documents(documents=crawl_1, ids=[d.metadata["id"] for d in crawl_1])
    print("Inserted ids:", inserted)

res = db.search_by_vector(db.dummy_vector, k=10)
print("Objects in the database:", len(res), res)
assert len(res) == 5, "Expected 4 objects in the database"

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
res = db.similarity_search_by_vector(db.dummy_vector, k=10)
print("Database objects after delete: ", len(res), res)
assert len(res) == 3, "Expected 3 objects in the database after deletion"

# Add new data
r = db.add_documents(data_add, ids=[d.metadata["id"] for d in data_add])
res = db.similarity_search_by_vector(db.dummy_vector, k=10)
print("Database objects after adding new", len(res), res)
assert len(res) == 5, "Expected 5 objects in the database after addition"

# Update data
db.update_last_seen_at(ids_update_meta)
res = db.search_by_vector(db.dummy_vector)
assert [r for r in res if r.metadata["id"] == "id3#3"][0].metadata["last_seen_at"] > 1, "Expected id3#3 to be updated"

res = db.search_by_vector(db.dummy_vector, k=10, filter_={"last_seen_at": {"$lt": 1}})
print("Expired objects in the database", len(res), res)
assert len(res) == 1, "Expected 1 expired object in the database"

db.delete_expired(expired_ts=1)

res = db.search_by_vector(db.dummy_vector)
print("Database objects after all updates", len(res), res)
assert len(res) == 4, "Expected 4 objects in the database after all updates"

# compare results with expected results
for expected in expected_results:
    d = [r for r in res if expected.metadata["id"] == r.metadata["id"]][0]
    assert d.metadata["item_id"] == expected.metadata["item_id"], f"Expected item_id {expected.metadata['item_id']}"
    assert d.metadata["checksum"] == expected.metadata["checksum"], f"Expected checksum {expected.metadata['checksum']}"

print("DONE")

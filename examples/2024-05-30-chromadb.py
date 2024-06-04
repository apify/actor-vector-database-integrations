import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from data_examples import compare_crawl_with_db, crawl_1, crawl_2, expected_results

load_dotenv(Path.cwd() / ".." / "code" / ".env")
CHROMA_COLLECTION_NAME = "apify"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True


chroma_client = chromadb.HttpClient(
    "localhost",
    port=8000,
    ssl=False,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=os.environ.get("CHROMA_SERVER_AUTHN_CREDENTIALS"),
    ),
)

print(chroma_client.heartbeat())
print(chroma_client.get_version())
print(chroma_client.count_collections())

vs_ = Chroma(client=chroma_client, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings)
index = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)


def wait_for_index(sec=1):
    time.sleep(sec)


if DROP_AND_INSERT:
    r = index.get()
    print("Objects in database", r)
    if r["ids"]:
        index.delete(ids=r["ids"])
        print("Deleted all objects from the database")

    # Insert objects
    inserted = vs_.add_documents(documents=crawl_1, ids=[d.metadata["id"] for d in crawl_1])
    print("Inserted ids:", inserted)
    print("Waiting for indexing")
    wait_for_index()

dummy_vector = embeddings.embed_query("a")

res = vs_.similarity_search_by_vector(dummy_vector, k=10)
print("Objects in the database:", len(res), res)
assert len(res) == 4, "Expected 4 objects in the database"

data_add, data_update_meta, data_del = compare_crawl_with_db(crawl_2, vs_.similarity_search_by_vector, dummy_vector)

print("Data to add", data_add)
print("Data to update", data_update_meta)
print("Data to delete", data_del)

assert len(data_add) == 2, "Expected 2 objects to add"
assert data_add[0].metadata["id"] == "id4#6"
assert data_add[1].metadata["id"] == "id5#5"

assert len(data_update_meta) == 1
assert data_update_meta[0].metadata["id"] == "id3#3"

assert len(data_del) == 1
assert data_del[0].metadata["id"] == "id4#4"

# Delete data that were updated
index.delete(ids=[d.metadata["id"] for d in data_del])
wait_for_index()
res = vs_.similarity_search_by_vector(dummy_vector, k=10)
print("Database objects after delete: ", len(res), res)
assert len(res) == 3, "Expected 3 objects in the database after deletion"

# Add new data
r = vs_.add_documents(data_add, ids=[d.metadata["id"] for d in data_add])
wait_for_index()
print("Added new crawled and updated objects", len(r), r)
res = vs_.similarity_search_by_vector(dummy_vector, k=10)
print("Database objects after adding new", len(res), res)
assert len(res) == 5, "Expected 5 objects in the database after addition"

# Update data
for d in data_update_meta:
    index.update(ids=[d.metadata["id"]], metadatas=[{"updated_at": d.metadata["updated_at"]}])
wait_for_index()

index.delete(where={"updated_at": {"$lt": 1}})
wait_for_index()

res = vs_.similarity_search_by_vector(dummy_vector, k=10000000)
print("Database objects after all updates", len(res), res)
assert len(res) == 4, "Expected 4 objects in the database after all updates"

# compare results with expected results
for expected in expected_results:
    d = [r for r in res if expected.metadata["id"] == r.metadata["id"]][0]
    assert d.metadata["item_id"] == expected.metadata["item_id"], f"Expected item_id {expected.metadata['item_id']}"
    assert d.metadata["checksum"] == expected.metadata["checksum"], f"Expected checksum {expected.metadata['checksum']}"
    assert d.metadata["updated_at"] == expected.metadata["updated_at"], f"Expected updated_at {expected.metadata['updated_at']}"

print("All tests passed")
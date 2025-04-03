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

from .data_examples_uuid import ITEM_ID1, crawl_1
from ..models.pinecone_input_model import PineconeIntegration
from ..vector_stores.pinecone import PineconeDatabase

load_dotenv()
PINECONE_INDEX_NAME = "apify"
PINECONE_INDEX_NAMESPACE = "ns1"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True

db = PineconeDatabase(
    actor_input=PineconeIntegration(
        pineconeIndexName=PINECONE_INDEX_NAME,
        pineconeIndexNamespace=PINECONE_INDEX_NAMESPACE,
        pineconeApiKey=os.getenv("PINECONE_API_KEY"),
        embeddingsProvider="OpenAI",
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)


def wait_for_index(sec=3):
    time.sleep(sec)


if DROP_AND_INSERT:
    r = list(db.index.list(prefix="", namespace=PINECONE_INDEX_NAMESPACE))
    print("Objects in database", r)
    if r:
        db.delete(ids=r)
        print("Deleted all objects from the database")

    # Insert objects
    inserted = db.add_documents(documents=crawl_1, ids=[d.metadata["chunk_id"] for d in crawl_1])
    print("Inserted ids:", inserted)
    print("Waiting for indexing")
    wait_for_index(10)
    print("Database stats", db.index.describe_index_stats())


res = db.search_by_vector(db.dummy_vector, k=10)
print("Objects in the database:", len(res), res)
assert len(res) == 6, "Expected 6 objects in the database"

print(db.get_by_item_id(ITEM_ID1))

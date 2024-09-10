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

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings

from .data_examples_uuid import crawl_1
from ..models.chroma_input_model import ChromaIntegration
from ..models.pinecone_input_model import EmbeddingsProvider
from ..vector_stores.chroma import ChromaDatabase

load_dotenv()
CHROMA_COLLECTION_NAME = "apify"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DROP_AND_INSERT = True

db = ChromaDatabase(
    ChromaIntegration(
        chromaCollectionName=CHROMA_COLLECTION_NAME,
        chromaClientHost="localhost",
        embeddingsProvider=EmbeddingsProvider.OpenAI.value,
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)
index = db.index

print("Database is connected: ", asyncio.run(db.is_connected()))


if DROP_AND_INSERT:
    db.delete_all()
    # Insert objects
    inserted = db.add_documents(documents=crawl_1, ids=[d.metadata["chunk_id"] for d in crawl_1])
    print("Inserted ids:", inserted)


r = db.similarity_search("text", k=100)
print("Search results:", r)
print("Search results count:", len(r))

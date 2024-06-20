# type: ignore
"""
This script serves as a playground for playing with Qdrant.

It demonstrates the process of performing delta updates on Qdrant. The process is as follows:
1. The database is initially populated with a set of crawled data (`crawl_1`).
2. A new set of data, `crawl_2`, is then crawled and compared with the existing data in the database.
3. The script contains several checks to validate that the database is updated correctly based on the comparison between `crawl_1` and `crawl_2`.

Run as a module:
    python -m src.examples.2024-06-20-Qdrant
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

from models.qdrant_input_model import QdrantIntegration
from ..models.pinecone_input_model import EmbeddingsProvider
from ..vector_stores.qdrant import QdrantDatabase

load_dotenv()
QDRANT_COLLECTION_NAME = "apify"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

DROP_AND_INSERT = True

db = QdrantDatabase(
    actor_input=QdrantIntegration(
        qdrantCollectionName=QDRANT_COLLECTION_NAME,
        qdrantUrl=os.getenv("QDRANT_URL"),
        qdrantApiKey=os.getenv("QDRANT_API_KEY"),
        embeddingsProvider=EmbeddingsProvider.OpenAI.value,
        embeddingsApiKey=os.getenv("OPENAI_API_KEY"),
        datasetFields=["text"],
    ),
    embeddings=embeddings,
)

r = db.add_documents([Document(page_content="dummy")], ids=["dummy"])

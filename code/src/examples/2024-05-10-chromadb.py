# type: ignore
"""
Example:
- Connect to the Chroma database (with or without authorization)
- Load data from Apify's dataset and store it in the Chroma database
- Perform a query search

Note: This example shows how to use the Actor SDK's Actor.open_dataset() method
directly, which is the recommended approach for Actors as it respects LIMITED_PERMISSIONS.

Run chroma docker:
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
"""

import asyncio
import os
import chromadb
from apify import Actor
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import create_dataset_mapping_function

async def main():
    COLLECTION_NAME = "apify"
    DATASET_ID = "uMZMKGf8lLA1u63FX"

    load_dotenv()

    # Load environment variables from .env file or specify them here
    CHROMA_HOST = os.getenv("CHROMA_CLIENT_HOST")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT")
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
    CHROMA_API_TOKEN = os.getenv("CHROMA_API_TOKEN")

    embeddings = OpenAIEmbeddings()

    chroma_client = chromadb.HttpClient(
        ssl=True, host=CHROMA_HOST, tenant=CHROMA_TENANT, database=CHROMA_DATABASE, headers={"x-chroma-token": CHROMA_API_TOKEN}
    )

    print(chroma_client.heartbeat())
    print(chroma_client.get_version())
    print(chroma_client.count_collections())

    # Use Actor SDK to load dataset (respects LIMITED_PERMISSIONS)
    async with Actor:
        mapping_function = create_dataset_mapping_function(fields=["text"], meta_object={}, meta_fields={})
        dataset = await Actor.open_dataset(name=DATASET_ID)

        data = []
        async for item in dataset.iterate_items():
            doc = mapping_function(item)
            if doc.page_content:
                data.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        data = text_splitter.split_documents(data)

        db = Chroma.from_documents(documents=data, client=chroma_client, embedding=embeddings, collection_name=COLLECTION_NAME)

        for v in db.similarity_search("apify", k=5):
            print(v)


if __name__ == "__main__":
    asyncio.run(main())

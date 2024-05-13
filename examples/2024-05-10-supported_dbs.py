import os
from functools import partial
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv(Path.cwd() / ".." / "code" / ".env")
COLLECTION_NAME = "chroma"

embeddings = OpenAIEmbeddings()

CHROMADB = False
PINECONE = True


docs = [
    Document(
        page_content="Apify is the platform where developers build, deploy, and "
        "publish web scraping, data extraction and web automation tools",
        metadata={"url": "https://apify.com"},
    )
]

# CHROMA --------------------
if CHROMADB:
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

    db = Chroma.from_documents(
        documents=docs, client=chroma_client, embedding=embeddings, collection_name=COLLECTION_NAME
    )
    print(db)

    for v in db.similarity_search("apify", k=1):
        print(v)

    # define partial function for Chroma.from_documents
    vs_from_documents = partial(Chroma.from_documents, client=chroma_client, collection_name=COLLECTION_NAME)
    vs_from_documents(documents=docs, embedding=embeddings)

# PINECONE --------------------
if PINECONE:
    # expects that PINECONE_API_KEY is set as the environment variable
    # db_p = PineconeVectorStore(embedding=embeddings)
    # client = PineconeClient(api_key=_pinecone_api_key, source_tag="langchain")
    # self._index = client.Index(_index_name)
    db_p = PineconeVectorStore.from_documents(
        documents=docs, embedding=embeddings, index_name=os.getenv("PINECONE_INDEX_NAME")
    )

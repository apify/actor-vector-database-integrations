import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv(Path.cwd() / ".." / "code" / ".env")
COLLECTION_NAME = "chroma"

embeddings = OpenAIEmbeddings()

CHROMADB = True
PINECONE = False


docs = [
    Document(
        page_content="Apify is the platform where developers build, deploy, and "
        "publish web scraping, data extraction and web automation tools",
        metadata={"url": "https://apify.com"},
    )
]

# CHROMA --------------------
if CHROMADB:

    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma

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

    vcs_ = Chroma(client=chroma_client, collection_name=COLLECTION_NAME, embedding_function=embeddings)
    vcs_.add_documents(documents=docs)
    for v in vcs_.similarity_search("apify", k=1):
        print(v)

# PINECONE --------------------
if PINECONE:
    from langchain_pinecone import PineconeVectorStore  # type: ignore
    from pinecone import Pinecone as PineconeClient  # type: ignore

    client = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"), source_tag="apify")
    vs_ = PineconeVectorStore(index=client.Index(os.getenv("PINECONE_INDEX_NAME")), embedding=embeddings)
    vs_.add_documents(documents=docs)

    for v in vs_.similarity_search("apify", k=1):
        print(v)

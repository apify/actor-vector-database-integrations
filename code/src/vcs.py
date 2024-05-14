import sys
from typing import TypeAlias

from apify import Actor
from langchain.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from .constants import PINECONE_SOURCE_TAG
from .models.chroma_input_model import ChromaIntegration
from .models.pinecone_input_model import PineconeIntegration

InputsDb: TypeAlias = ChromaIntegration | PineconeIntegration


async def get_vector_store(aid: InputsDb, embeddings: Embeddings) -> VectorStore:
    """Get database based on the integration type."""

    if isinstance(aid, ChromaIntegration):
        return await _get_chroma(aid, embeddings)

    if isinstance(aid, PineconeIntegration):
        return await _get_pinecone(aid, embeddings)

    await Actor.fail(status_message=f"Failed to get database with config: {aid}")
    raise ValueError("Failed to get database")


async def _get_chroma(aid: ChromaIntegration, embeddings: Embeddings) -> VectorStore:

    # Apify dockerfile is using Debian slim-buster image, which has an unsupported version of sqlite3.
    # FIx for RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
    # References:
    #  https://docs.trychroma.com/troubleshooting#sqlite
    #  https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
    #
    # pip install pysqlite3
    # swap the stdlib sqlite3 lib with the pysqlite3 package, before importing chromadb
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

    import chromadb
    from langchain_chroma import Chroma

    settings = None
    if auth := aid.chroma_server_auth_credentials:
        settings = chromadb.config.Settings(
            chroma_client_auth_credentials=auth,
            chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        )
    try:
        chroma_client = chromadb.HttpClient(
            host=aid.chroma_client_host,
            port=aid.chroma_client_port or 8000,
            ssl=aid.chroma_client_ssl or False,
            settings=settings,
        )
        assert chroma_client.heartbeat() > 1
        Actor.log.debug("Connected to chroma database")
        return Chroma(
            client=chroma_client, collection_name=aid.chroma_collection_name or "chroma", embedding_function=embeddings
        )

    except Exception as e:
        await Actor.fail(status_message=f"Failed to connect to chroma: {e}")
        raise


async def _get_pinecone(aid: PineconeIntegration, embeddings: Embeddings) -> VectorStore:

    from langchain_pinecone import PineconeVectorStore  # type: ignore
    from pinecone import Pinecone as PineconeClient  # type: ignore

    try:
        client = PineconeClient(api_key=aid.pinecone_api_key, source_tag=PINECONE_SOURCE_TAG)
        return PineconeVectorStore(index=client.Index(aid.pinecone_index_name), embedding=embeddings)
    except Exception as e:
        await Actor.fail(status_message=f"Failed to initialize pinecone: {e}")
        raise

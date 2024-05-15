from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TypeAlias

from apify import Actor

from .constants import PINECONE_SOURCE_TAG
from .exceptions import FailedToConnectToDatabaseError
from .models.chroma_input_model import ChromaIntegration
from .models.pinecone_input_model import PineconeIntegration

if TYPE_CHECKING:
    from langchain.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings

InputsDb: TypeAlias = ChromaIntegration | PineconeIntegration


async def get_vector_store(aid: InputsDb | None, embeddings: Embeddings) -> VectorStore:
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

    if TYPE_CHECKING:
        from chromadb.api import ClientAPI
    async def check_chroma_connection(client: ClientAPI) -> None:
        if client.heartbeat() <= 1:
            raise FailedToConnectToDatabaseError("ChromaDB is not reachable")
        Actor.log.debug("Connected to chroma database")

    settings = None
    if auth := aid.chromaServerAuthCredentials:
        settings = chromadb.config.Settings(
            chroma_client_auth_credentials=auth,
            chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        )
    try:
        chroma_client = chromadb.HttpClient(
            host=aid.chromaClientHost,
            port=aid.chromaClientPort or 8000,
            ssl=aid.chromaClientSsl or False,
            settings=settings,
        )
        await check_chroma_connection(chroma_client)
        return Chroma(
            client=chroma_client, collection_name=aid.chromaCollectionName or "chroma", embedding_function=embeddings
        )

    except Exception as e:
        await Actor.fail(status_message=f"Failed to connect to chroma: {e}")
        raise FailedToConnectToDatabaseError("Failed to connect to chroma") from e


async def _get_pinecone(aid: PineconeIntegration, embeddings: Embeddings) -> VectorStore:

    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]

    try:
        client = PineconeClient(api_key=aid.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
        return PineconeVectorStore(index=client.Index(aid.pineconeIndexName), embedding=embeddings)
    except Exception as e:
        await Actor.fail(status_message=f"Failed to initialize pinecone: {e}")
        raise FailedToConnectToDatabaseError("Failed to connect to chroma") from e

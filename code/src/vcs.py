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

ActorInputsDb: TypeAlias = ChromaIntegration | PineconeIntegration


async def get_vector_store(actor_input: ActorInputsDb | None, embeddings: Embeddings) -> VectorStore:
    """Get database based on the integration type."""

    if isinstance(actor_input, ChromaIntegration):
        return await _get_chroma(actor_input, embeddings)

    if isinstance(actor_input, PineconeIntegration):
        return await _get_pinecone(actor_input, embeddings)

    await Actor.fail(status_message=f"Failed to get database with config: {actor_input}")
    raise ValueError("Failed to get database")


async def _get_chroma(actor_input: ChromaIntegration, embeddings: Embeddings) -> VectorStore:
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
        Actor.log.info("Connected to chroma database")

    settings = None
    if auth := actor_input.chromaServerAuthCredentials:
        settings = chromadb.config.Settings(
            chroma_client_auth_credentials=auth,
            chroma_client_auth_provider=actor_input.chromaClientAuthProvider,
        )
    try:
        chroma_client = chromadb.HttpClient(
            host=actor_input.chromaClientHost,
            port=actor_input.chromaClientPort or 8000,
            ssl=actor_input.chromaClientSsl or False,
            settings=settings,
        )
        await check_chroma_connection(chroma_client)
        return Chroma(
            client=chroma_client,
            collection_name=actor_input.chromaCollectionName or "chroma",
            embedding_function=embeddings,
        )

    except Exception as e:
        await Actor.fail(status_message=f"Failed to connect to chroma: {e}")
        raise FailedToConnectToDatabaseError("Failed to connect to chroma") from e


async def _get_pinecone(actor_input: PineconeIntegration, embeddings: Embeddings) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]

    try:
        client = PineconeClient(api_key=actor_input.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
        return PineconeVectorStore(index=client.Index(actor_input.pineconeIndexName), embedding=embeddings)
    except Exception as e:
        await Actor.fail(status_message=f"Failed to initialize pinecone: {e}")
        raise FailedToConnectToDatabaseError("Failed to connect to chroma") from e

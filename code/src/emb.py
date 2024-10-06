from __future__ import annotations

from typing import TYPE_CHECKING

from apify import Actor

from .constants import SupportedEmbeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


async def get_embedding_provider(embeddings_name: str, api_key: str | None = None, config: dict | None = None) -> Embeddings:
    """Return the embeddings based on the user preference."""

    if embeddings_name == SupportedEmbeddings.openai:
        from langchain_openai.embeddings import OpenAIEmbeddings

        config = config or {}
        config["openai_api_key"] = api_key
        return config and OpenAIEmbeddings(**config) or OpenAIEmbeddings()

    if embeddings_name == SupportedEmbeddings.cohere:
        from langchain_cohere import CohereEmbeddings

        config = config or {}
        config["cohere_api_key"] = api_key
        return CohereEmbeddings(**config)

    if embeddings_name == SupportedEmbeddings.fake:
        from langchain_core.embeddings import FakeEmbeddings

        config = config or {}
        return FakeEmbeddings(**config)

    await Actor.fail(status_message=f"Failed to get embeddings for embeddings: {embeddings_name} and config: {config}")
    raise ValueError("Failed to get embeddings")

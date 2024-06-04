from __future__ import annotations

from typing import TYPE_CHECKING

from apify import Actor

from .constants import SupportedEmbeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


async def get_embeddings(embeddings_class_name: str, api_key: str | None = None, config: dict | None = None) -> Embeddings:
    """Return the embeddings based on the user preference."""

    if embeddings_class_name == SupportedEmbeddings.open_ai_embeddings:
        from langchain_openai.embeddings import OpenAIEmbeddings

        config = config or {}
        config["openai_api_key"] = api_key
        return config and OpenAIEmbeddings(**config) or OpenAIEmbeddings()

    if embeddings_class_name == SupportedEmbeddings.cohere_embeddings:
        from langchain_cohere import CohereEmbeddings

        config = config or {}
        config["cohere_api_key"] = api_key
        return CohereEmbeddings(**config)

    await Actor.fail(status_message=f"Failed to get embeddings for embedding class: {embeddings_class_name} and config: {config}")
    raise ValueError("Failed to get embeddings")

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from apify import Actor

from .constants import SupportedEmbeddingsEn

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


async def get_embeddings(embeddings_class: str, api_key: str | None = None, config: dict | None = None) -> Embeddings:
    """Return the embeddings based on the user preference."""

    if embeddings_class == SupportedEmbeddingsEn.open_ai_embeddings:
        from langchain_openai.embeddings import OpenAIEmbeddings

        os.environ["OPENAI_API_KEY"] = api_key or ""
        return config and OpenAIEmbeddings(**config) or OpenAIEmbeddings()

    if embeddings_class == SupportedEmbeddingsEn.cohere_embeddings and config:
        from langchain_cohere import CohereEmbeddings

        config["cohere_api_key"] = api_key
        return CohereEmbeddings(**config)

    if embeddings_class == SupportedEmbeddingsEn.hugging_face_embeddings and config:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(**config)

    await Actor.fail(
        status_message=f"Failed to get embeddings for embedding class: {embeddings_class} and config: {config}"
    )
    raise ValueError("Failed to get embeddings")

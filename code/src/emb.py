import os
from typing import Literal, TypeAlias

from apify import Actor
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings

from .models.chroma_input_model import ChromaIntegration
from .models.pinecone_input_model import PineconeIntegration

InputsDb: TypeAlias = ChromaIntegration | PineconeIntegration


SupportedEmbeddings = Literal["OpenAIEmbeddings", "CohereEmbeddings", "HuggingFaceEmbeddings"]


async def get_embeddings(
    embeddings_class: SupportedEmbeddings, api_key: str | None = None, config: dict | None = None
) -> Embeddings:
    """Return the embeddings based on the user preference."""

    if embeddings_class == "OpenAIEmbeddings":
        os.environ["OPENAI_API_KEY"] = api_key or ""
        return config and OpenAIEmbeddings(**config) or OpenAIEmbeddings()

    if embeddings_class == "CohereEmbeddings" and config:
        config["cohere_api_key"] = api_key
        return CohereEmbeddings(**config)

    if embeddings_class == "HuggingFaceEmbeddings" and config:
        return HuggingFaceEmbeddings(**config)

    await Actor.fail(status_message=f"Failed to get embeddings for: {embeddings_class} and config: {config}")
    raise ValueError("Failed to get embeddings")

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .models.chroma_input_model import ChromaIntegration
    from .models.pinecone_input_model import PineconeIntegration
    from .models.qdrant_input_model import QdrantIntegration
    from .vector_stores.chroma import ChromaDatabase
    from .vector_stores.pinecone import PineconeDatabase
    from .vector_stores.qdrant import QdrantDatabase

    ActorInputsDb: TypeAlias = ChromaIntegration | PineconeIntegration | QdrantIntegration
    VectorDb: TypeAlias = ChromaDatabase | PineconeDatabase | QdrantDatabase

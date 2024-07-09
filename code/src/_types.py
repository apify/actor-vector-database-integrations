from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .models import ChromaIntegration, MilvusIntegration, PgvectorIntegration, PineconeIntegration, QdrantIntegration, WeaviateIntegration
    from .vector_stores import ChromaDatabase, MilvusDatabase, PGVectorDatabase, PineconeDatabase, QdrantDatabase, WeaviateDatabase

    ActorInputsDb: TypeAlias = (
        ChromaIntegration | MilvusIntegration | PgvectorIntegration | PineconeIntegration | QdrantIntegration | WeaviateIntegration
    )
    VectorDb: TypeAlias = ChromaDatabase | MilvusDatabase | PGVectorDatabase | PineconeDatabase | QdrantDatabase | WeaviateDatabase

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .models import (
        ChromaIntegration,
        MilvusIntegration,
        PgvectorIntegration,
        PineconeIntegration,
        QdrantIntegration,
        VectorxIntegration,
        WeaviateIntegration,
    )
    from .vector_stores import (
        ChromaDatabase,
        MilvusDatabase,
        PGVectorDatabase,
        PineconeDatabase,
        QdrantDatabase,
        VectorxDatabase,
        WeaviateDatabase,
    )

    ActorInputsDb: TypeAlias = (
        ChromaIntegration
        | MilvusIntegration
        | PgvectorIntegration
        | PineconeIntegration
        | QdrantIntegration
        | VectorxIntegration
        | WeaviateIntegration
    )
    VectorDb: TypeAlias = ChromaDatabase | MilvusDatabase | PGVectorDatabase | PineconeDatabase | QdrantDatabase | VectorxDatabase | WeaviateDatabase

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .models import ChromaIntegration, PgvectorIntegration, PineconeIntegration, QdrantIntegration
    from .vector_stores import ChromaDatabase, PGVectorDatabase, PineconeDatabase, QdrantDatabase

    ActorInputsDb: TypeAlias = ChromaIntegration | PgvectorIntegration | PineconeIntegration | QdrantIntegration
    VectorDb: TypeAlias = ChromaDatabase | PGVectorDatabase | PineconeDatabase | QdrantDatabase

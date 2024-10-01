import os

from apify import Actor

from .constants import SupportedVectorStores
from .main import run_actor
from .models import (
    ChromaIntegration,
    MilvusIntegration,
    OpensearchIntegration,
    PgvectorIntegration,
    PineconeIntegration,
    QdrantIntegration,
    WeaviateIntegration,
)


async def main() -> None:
    async with Actor:
        Actor.log.info("Starting the Vector Store Actor")

        if not (actor_input := await Actor.get_input() or {}):
            await Actor.fail(status_message="No input provided", exit_code=1)

        if not (arg := os.getenv("ACTOR_PATH_IN_DOCKER_CONTEXT")):
            if Actor.is_at_home():
                await Actor.exit(
                    exit_code=100,
                    status_message="This Actor was built incorrectly; no environment variable specifies which Actor "
                    "to start. If you encounter this issue, please contact the Actor developer.",
                )

            arg = f"actors/{SupportedVectorStores.pinecone.value}"
            Actor.log.warning(
                f"The environment variable ACTOR_PATH_IN_DOCKER_CONTEXT was not specified. " f"Using default for local development: {arg}"
            )

        actor_type = arg.split("/")[-1]
        Actor.log.info("Received start argument (vector database name): %s", actor_type)

        if actor_type == SupportedVectorStores.chroma.value:
            await run_actor(ChromaIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStores.milvus.value:
            await run_actor(MilvusIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStores.opensearch.value:
            await run_actor(OpensearchIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStores.pgvector.value:
            await run_actor(PgvectorIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStores.pinecone.value:
            await run_actor(PineconeIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStores.qdrant.value:
            await run_actor(QdrantIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStores.weaviate.value:
            await run_actor(WeaviateIntegration(**actor_input), actor_input)
        else:
            await Actor.exit(
                exit_code=10,
                status_message=f"This Actor was built incorrectly; an unknown Actor was selected "
                f"to start ({actor_type}). If you encounter this issue, please contact the Actor developer.",
            )

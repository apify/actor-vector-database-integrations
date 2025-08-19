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

        # Set the Apify API token if it is available in the environment variables.
        if apify_api_token := os.getenv("APIFY_TOKEN"):
            os.environ["APIFY_API_TOKEN"] = apify_api_token

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

        actor_input_ensure_backward_compatibility(actor_input)
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


def actor_input_ensure_backward_compatibility(actor_input: dict) -> None:
    """Ensure backward compatibility for the actor input."""
    if not actor_input.get("dataUpdatesStrategy"):
        # legacy update mechanism
        if actor_input.get("enableDeltaUpdates") is False:
            actor_input["dataUpdatesStrategy"] = "add"
        else:
            actor_input["dataUpdatesStrategy"] = "deltaUpdates"
    else:
        # for integrations that do not have updateStrategy implemented
        actor_input["enableDeltaUpdates"] = actor_input["dataUpdatesStrategy"] == "deltaUpdates"

    if not actor_input.get("dataUpdatesPrimaryDatasetFields"):
        actor_input["dataUpdatesPrimaryDatasetFields"] = actor_input.get("deltaUpdatesPrimaryDatasetFields", [])
    else:
        # for integrations that do not have updateStrategy implemented
        actor_input["deltaUpdatesPrimaryDatasetFields"] = actor_input.get("dataUpdatesPrimaryDatasetFields")

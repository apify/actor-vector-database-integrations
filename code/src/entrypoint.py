import os

from apify import Actor

from .constants import SupportedVectorStoresEn
from .main import main as main_f
from .models.chroma_input_model import ChromaIntegration
from .models.pinecone_input_model import PineconeIntegration


async def main():
    async with Actor:

        Actor.log.info("Starting the Vector Store Actor")

        if not (actor_input := await Actor.get_input() or {}):
            await Actor.fail(status_message="No input provided", exit_code=1)

        Actor.log.info("Received input")

        Actor.log.info(
            "Checking for the environment variable ACTOR_PATH_IN_DOCKER_CONTEXT: %s",
            os.getenv("ACTOR_PATH_IN_DOCKER_CONTEXT"),
        )
        arg = os.getenv("ACTOR_PATH_IN_DOCKER_CONTEXT")

        Actor.log.info("arg: %s", arg)

        if not arg:
            Actor.log.info("No environment variable ACTOR_PATH_IN_DOCKER_CONTEXT found")
            if Actor.is_at_home():
                Actor.log.info("Running in local development mode")
                await Actor.exit(
                    exit_code=100,
                    status_message="This Actor was built incorrectly; no environment variable specifies which Actor "
                    "to start. If you encounter this issue, please contact the Actor developer.",
                )

            Actor.log.info("Running in Apify cloud")
            arg = f"actors/{SupportedVectorStoresEn.pinecone.value}"
            Actor.log.warning(
                f"The environment variable ACTOR_PATH_IN_DOCKER_CONTEXT was not specified. "
                f"Using default for local development: actors/{arg}"
            )

        Actor.log.info("Actor path: %s", arg)
        actor_type = arg.split("/")[-1]
        Actor.log.info("Received start argument: %s", actor_type)

        if actor_type == SupportedVectorStoresEn.chroma.value:
            return main_f(ChromaIntegration(**actor_input), actor_input)
        elif actor_type == SupportedVectorStoresEn.pinecone.value:
            return main_f(PineconeIntegration(**actor_input), actor_input)
        else:
            await Actor.exit(
                exit_code=10,
                status_message=f"This Actor was built incorrectly; an unknown Actor was selected to start ({actor_type}). "
                f"If you encounter this issue, please contact the Actor developer.",
            )
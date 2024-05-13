import os

from apify import Actor
from langchain.docstore.document import Document
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vcs import TypeDb, get_database
from .models.chroma_input_model import ChromaIntegration  # type: ignore
from .models.pinecone_input_model import PineconeIntegration  # type: ignore


def get_nested_value(data: dict, keys: str) -> str:
    """
    Extract nested value from dict.

    Example:
      >>> get_nested_value({"a": "v1", "c1": {"c2": "v2"}}, "c1.c2")
      'v2'
    """

    keys_list = keys.split(".")
    result = data

    for key in keys_list:
        if key in result:
            result = result[key]
        else:
            # If any of the keys are not found, return None
            return ""

    return result  # type: ignore


async def main():
    async with Actor:

        if not (actor_input := await Actor.get_input() or {}):
            await Actor.fail(status_message="No input provided", exit_code=1)

        Actor.log.debug("Received start argument: %s", actor_input)

        os.environ["OPENAI_API_KEY"] = actor_input.openai_api_key

        resource = payload.get("payload", {}).get("resource", {})
        if not (dataset_id := resource.get("defaultDatasetId") or payload.get("dataset_id", "")):
            msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
            await Actor.fail(status_message=msg)

        Actor.log.debug("Load Dataset ID %s and extract fields %s", dataset_id, actor_input.fields)

        embeddings = OpenAIEmbeddings()

        meta_values = actor_input.metadata_values or {}
        meta_fields = actor_input.metadata_fields or {}

        # Function from Honza Turon to load dataset.
        # Do we really want to create a new chunk for every field?
        for field in actor_input.fields:
            loader = ApifyDatasetLoader(
                dataset_id,
                dataset_mapping_function=lambda dataset_item: Document(
                    page_content=get_nested_value(dataset_item, field),
                    metadata={
                        **meta_values,
                        **{key: get_nested_value(dataset_item, value) for key, value in meta_fields.items()},
                    },
                ),
            )

            try:
                documents = loader.load()
                Actor.log.debug("Document loaded")
            except Exception as e:
                await Actor.fail(status_message=f"Failed to load documents for field {field}: {e}")

            if actor_input.perform_chunking:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=actor_input.chunk_size, chunk_overlap=actor_input.chunk_overlap
                )
                documents = text_splitter.split_documents(documents)
                Actor.log.debug("Documents chunked to %s chunks", len(documents))

            try:
                pf_from_documents = await get_database(actor_input)
                pf_from_documents(documents=documents, embedding=embeddings)
                Actor.log.debug("Documents inserted into database successfully")
            except Exception as e:
                msg = f"Document insertion failed: {str(e)}"
                await Actor.set_status_message(msg)
                await Actor.fail()

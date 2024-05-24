from typing import TYPE_CHECKING

from apify import Actor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .emb import get_embeddings
from .utils import get_dataset_loader
from .vcs import ActorInputsDb, get_vector_store

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore


async def run_actor(actor_input: ActorInputsDb, payload: dict) -> None:

    resource = payload.get("payload", {}).get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or actor_input.datasetId):
        msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
        await Actor.fail(status_message=msg)

    try:
        Actor.log.info("Get embeddings class: %s", actor_input.embeddings.value)  # type: ignore[union-attr]
        embeddings = await get_embeddings(
            actor_input.embeddings.value,  # type: ignore[union-attr]
            actor_input.embeddingsApiKey,
            actor_input.embeddingsConfig,
        )
    except Exception as e:
        msg = f"Failed to get embeddings: {e}"
        await Actor.fail(status_message=msg)
        return

    Actor.log.info("Load Dataset ID %s and extract fields %s", dataset_id, actor_input.fields)
    try:
        dataset_loader = get_dataset_loader(
            str(actor_input.datasetId),
            fields=actor_input.fields,
            meta_values=actor_input.metadataValues or {},
            meta_fields=actor_input.metadataFields or {},
        )
        documents = dataset_loader.load()
        documents = [doc for doc in documents if doc.page_content]
        Actor.log.info("Dataset loaded, number of documents: %s", len(documents))
    except Exception as e:
        await Actor.fail(status_message=f"Failed to load datasets: {e}")
        return

    if actor_input.performChunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=actor_input.chunkSize, chunk_overlap=actor_input.chunkOverlap)
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))
    try:
        vcs_: VectorStore = await get_vector_store(actor_input, embeddings)
        vcs_.add_documents(documents)
        Actor.log.info("Documents inserted into database successfully")

        Actor.log.info("Push chunked data to the unnamed output dataset")
        await Actor.push_data([doc.dict() for doc in documents])
    except Exception as e:
        await Actor.set_status_message(f"Document insertion failed: {e}")
        await Actor.fail()

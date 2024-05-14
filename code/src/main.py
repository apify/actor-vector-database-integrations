from apify import Actor
from langchain.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from emb import SupportedEmbeddings
from .emb import get_embeddings
from .utils import load_dataset
from .vcs import InputsDb, get_vector_store


async def main(actor_input: InputsDb, payload: dict):

    resource = payload.get("payload", {}).get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or actor_input.dataset_id):
        msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
        await Actor.fail(status_message=msg)

    Actor.log.debug("Load Dataset ID %s and extract fields %s", dataset_id, actor_input.fields)

    embeddings = await get_embeddings(
        SupportedEmbeddings(actor_input.embeddings), actor_input.embeddings_api_key, actor_input.embeddings_config
    )

    try:
        loader_ = load_dataset(
            str(actor_input.dataset_id),
            fields=actor_input.fields,
            meta_values=actor_input.metadata_values or {},
            meta_fields=actor_input.metadata_fields or {},
        )
        documents = loader_.load()
        Actor.log.info("Datasets loaded")
    except Exception as e:
        await Actor.fail(status_message=f"Failed to load datasets: {e}")
        return

    if actor_input.perform_chunking:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=actor_input.chunk_size, chunk_overlap=actor_input.chunk_overlap
        )
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))
    try:
        vcs_: VectorStore = await get_vector_store(actor_input, embeddings)
        vcs_.add_documents(documents)
        Actor.log.info("Documents inserted into database successfully")
    except Exception as e:
        msg = f"Document insertion failed: {str(e)}"
        await Actor.set_status_message(msg)
        await Actor.fail()

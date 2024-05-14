from apify import Actor
from langchain.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .emb import get_embeddings
from .utils import load_dataset
from .vcs import InputsDb, get_vector_store


async def main(aid: InputsDb, payload: dict):

    resource = payload.get("payload", {}).get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or aid.dataset_id):
        msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
        await Actor.fail(status_message=msg)

    try:
        Actor.log.info("Getting embeddings: %s", aid.embeddings.value)  # type: ignore
        embeddings = await get_embeddings(
            aid.embeddings.value, aid.embeddings_api_key, aid.embeddings_config  # type: ignore
        )
    except Exception as e:
        msg = f"Failed to get embeddings: {str(e)}"
        await Actor.fail(status_message=msg)
        return

    Actor.log.info("Load Dataset ID %s and extract fields %s", dataset_id, aid.fields)
    try:
        loader_ = load_dataset(
            str(aid.dataset_id),
            fields=aid.fields,
            meta_values=aid.metadata_values or {},
            meta_fields=aid.metadata_fields or {},
        )
        documents = loader_.load()
        documents = [doc for doc in documents if doc.page_content]
        Actor.log.info("Dataset loaded, number of documents: %s", len(documents))
    except Exception as e:
        await Actor.fail(status_message=f"Failed to load datasets: {e}")
        return

    if aid.perform_chunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=aid.chunk_size, chunk_overlap=aid.chunk_overlap)
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))
    try:
        vcs_: VectorStore = await get_vector_store(aid, embeddings)
        vcs_.add_documents(documents)
        Actor.log.info("Documents inserted into database successfully")
    except Exception as e:
        msg = f"Document insertion failed: {str(e)}"
        await Actor.set_status_message(msg)
        await Actor.fail()

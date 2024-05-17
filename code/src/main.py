from typing import TYPE_CHECKING

from apify import Actor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .emb import get_embeddings
from .utils import load_dataset
from .vcs import InputsDb, get_vector_store

if TYPE_CHECKING:
    from langchain.vectorstores import VectorStore


async def main(aid: InputsDb, payload: dict) -> None:
    resource = payload.get("payload", {}).get("resource", {})
    if not (dataset_id := resource.get("defaultDatasetId") or aid.datasetId):
        msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
        await Actor.fail(status_message=msg)

    try:
        Actor.log.info("Get embeddings class: %s", aid.embeddings.value)  # type: ignore[union-attr]
        embeddings = await get_embeddings(
            aid.embeddings.value,  # type: ignore[union-attr]
            aid.embeddingsApiKey,
            aid.embeddingsConfig,
        )
    except Exception as e:
        msg = f"Failed to get embeddings: {e}"
        await Actor.fail(status_message=msg)
        return

    Actor.log.info("Load Dataset ID %s and extract fields %s", dataset_id, aid.fields)
    try:
        loader_ = load_dataset(
            str(aid.datasetId),
            fields=aid.fields,
            meta_values=aid.metadataValues or {},
            meta_fields=aid.metadataFields or {},
        )
        documents = loader_.load()
        documents = [doc for doc in documents if doc.page_content]
        Actor.log.info("Dataset loaded, number of documents: %s", len(documents))
    except Exception as e:
        await Actor.fail(status_message=f"Failed to load datasets: {e}")
        return

    if aid.performChunking:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=aid.chunkSize, chunk_overlap=aid.chunkOverlap)
        documents = text_splitter.split_documents(documents)
        Actor.log.info("Documents chunked to %s chunks", len(documents))
    try:
        vcs_: VectorStore = await get_vector_store(aid, embeddings)
        vcs_.add_documents(documents)
        Actor.log.info("Documents inserted into database successfully")

        Actor.log.info("Push chunked data to the unnamed output dataset")
        await Actor.push_data([doc.dict() for doc in documents])
    except Exception as e:
        msg = f"Document insertion failed: {e}"
        await Actor.set_status_message(msg)
        await Actor.fail()

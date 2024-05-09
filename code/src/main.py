import os
import sys


# Apify dockerfile is using Debian slim-buster image, which has an unsupported version of sqlite3.
# FIx for RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
# References:
#  https://docs.trychroma.com/troubleshooting#sqlite
#  https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300
#
# pip install pysqlite3
# swap the stdlib sqlite3 lib with the pysqlite3 package, before importing chromadb
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from apify import Actor
import chromadb
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_nested_value(data: dict, keys: str):
    """
    Extract nested value from dict.

    Example:
      >>> get_nested_value({"a": "v1", "c1": {"c2": "v2"}}, "c1.c2")
      'v2'
    """

    keys = keys.split(".")
    result = data

    for key in keys:
        if key in result:
            result = result[key]
        else:
            # If any of the keys are not found, return None
            return None

    return result


async def try_except(success, failure):
    try:
        return success()
    except Exception:
        await Actor.fail(status_message=failure)


async def main():
    async with Actor:

        # Get the value of the actor input
        actor_input = await Actor.get_input() or {}

        chroma_coll_name = actor_input.get("chroma_collection_name")
        chroma_client_host = actor_input.get("chroma_client_host") or ""
        chroma_client_port = actor_input.get("chroma_client_port")
        chroma_client_ssl = actor_input.get("chroma_client_ssl")

        settings = None
        if auth := actor_input.get("chroma_server_auth_credentials"):
            settings = chromadb.config.Settings(
                chroma_client_auth_credentials=auth,
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
            )

        os.environ["OPENAI_API_KEY"] = actor_input.get("openai_token") or ""

        fields = actor_input.get("fields") or []
        metadata_fields = actor_input.get("metadata_fields") or {}
        metadata_values = actor_input.get("metadata_values") or {}

        perform_chunking = actor_input.get("perform_chunking")
        chunk_size, chunk_overlap = actor_input.get("chunk_size"), actor_input.get("chunk_overlap")

        resource = actor_input.get("payload", {}).get("resource", {})

        if not (dataset_id := resource.get("defaultDatasetId") or actor_input.get("dataset_id")):
            msg = "No Dataset ID provided. It should be provided either in payload or in actor_input"
            await Actor.fail(status_message=msg)

        try:
            chroma_client = chromadb.HttpClient(
                host=chroma_client_host, port=chroma_client_port, ssl=chroma_client_ssl, settings=settings
            )
            assert chroma_client.heartbeat() > 1
            Actor.log.debug("Connected to chroma")
        except Exception as e:
            msg = f"Failed to connect to chroma: {str(e)}"
            await Actor.fail(status_message=msg)

        Actor.log.debug("Load Dataset ID %s and extract fields %s", dataset_id, fields)

        embeddings = OpenAIEmbeddings()

        for field in fields:
            loader = ApifyDatasetLoader(
                dataset_id,
                dataset_mapping_function=lambda dataset_item: Document(
                    page_content=get_nested_value(dataset_item, field),
                    metadata={
                        **metadata_values,
                        **{key: get_nested_value(dataset_item, value) for key, value in metadata_fields.items()},
                    },
                ),
            )

            documents = await try_except(loader.load, f"Failed to load documents for field {field}")
            Actor.log.debug("Document loaded")

            if perform_chunking:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                documents = text_splitter.split_documents(documents)
                Actor.log.debug("Documents chunked to %s chunks", len(documents))

            try:
                Chroma.from_documents(
                    documents=documents, client=chroma_client, embedding=embeddings, collection_name=chroma_coll_name
                )
                Actor.log.debug("Documents inserted into ChromaDB")
            except Exception as e:
                msg = f"Document insertion into ChromaDB failed: {str(e)}"
                await Actor.set_status_message(msg)
                await Actor.fail()

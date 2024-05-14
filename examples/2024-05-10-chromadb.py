import chromadb
from apify_client._errors import ApifyApiError
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from apify_client import ApifyClient

# docker pull chromadb/chroma
# docker run -p 8000:8000 chromadb/chroma

COLLECTION_NAME = "chroma"
DATASET_ID = "J13oTZVY5wJWQdbzn"


load_dotenv(Path.cwd() / ".." / "code" / ".env")

embeddings = OpenAIEmbeddings()

client = ApifyClient()

chroma_client = chromadb.HttpClient(
    "localhost",
    port=8000,
    ssl=False,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials="test-token",
    ),
)

print(chroma_client.heartbeat())
print(chroma_client.get_version())
print(chroma_client.count_collections())


def get_nested_value(d: dict, keys: str) -> str:
    """
    Extract nested value from dict.

    Example:
      >>> get_nested_value({"a": "v1", "c1": {"c2": "v2"}}, "c1.c2")
      'v2'
    """

    keys_list = keys.split(".")
    result = d

    for key in keys_list:
        if key in result:
            result = result[key]
        else:
            # If any of the keys are not found, return None
            return ""

    return result  # type: ignore




def load_dataset(dataset_id: str) -> ApifyDatasetLoader:
    """Load dataset by dataset_id using ApifyDatasetLoader."""

    try:
        return ApifyDatasetLoader(
            dataset_id=dataset_id,
            dataset_mapping_function=lambda dataset_item: Document(
                page_content=dataset_item["text"] or "", metadata={"source": dataset_item["url"]}
            ),
        )
    except ApifyApiError as e:
        raise ApifyApiError(e)

dataset = client.dataset(DATASET_ID).list_items(clean=True)
data: list = dataset.items
data = [{key: get_nested_value(d, key) for key in aid.fields} for d in data]
data = [d for d in data if d]

# loader: ApifyDatasetLoader = load_dataset(DATASET_ID)
# data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
data = text_splitter.split_documents(data)

db = Chroma.from_documents(documents=data, client=chroma_client, embedding=embeddings, collection_name=COLLECTION_NAME)
print(db)

for v in db.similarity_search("apify", k=5):
    print(v)

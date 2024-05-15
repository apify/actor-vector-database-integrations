import sys
from pathlib import Path

import chromadb
from apify_client import ApifyClient
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "src"))

from utils import get_nested_value, load_dataset, stringify_dict  # type: ignore

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

loader: ApifyDatasetLoader = load_dataset(DATASET_ID, fields=["text"], meta_values={}, meta_fields={})
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
data = text_splitter.split_documents(data)

db = Chroma.from_documents(documents=data, client=chroma_client, embedding=embeddings, collection_name=COLLECTION_NAME)

for v in db.similarity_search("apify", k=5):
    print(v)

# type: ignore
"""
Example"
- Connect to the Chroma database (with or without authorization)
- Load data from Apify's dataset and store it in the Chroma database
- Perform a query search

Run chroma docker

docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
"""

import chromadb
from apify_client import ApifyClient
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import get_dataset_loader

COLLECTION_NAME = "chroma"
DATASET_ID = "u3lJ4arH2sC28ch3i"


load_dotenv()

embeddings = OpenAIEmbeddings()

client = ApifyClient()

chroma_client = chromadb.HttpClient(
    host="localhost",
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

loader: ApifyDatasetLoader = get_dataset_loader(DATASET_ID, fields=["text"], meta_object={}, meta_fields={})
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
data = text_splitter.split_documents(data)

db = Chroma.from_documents(documents=data, client=chroma_client, embedding=embeddings, collection_name=COLLECTION_NAME)

for v in db.similarity_search("apify", k=5):
    print(v)

# type: ignore
"""
Apify ChromaDB integration
Use Apify's ChromaDB integration to transfer website content data into a ChromaDB database.

This example uses the Website Content Crawler to crawl the ChromaDB documentation website and uploads the data to a ChromaDB database.

pip install apify-client
"""

import os

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv("../../.env")

# Load environment variables from .env file or specify them here
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN") or "YOUR-APIFY-TOKEN"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY"
CHROMA_HOST = os.getenv("CHROMA_CLIENT_HOST") or "YOUR-CHROMA-HOST"
CHROMA_API_TOKEN = os.getenv("CHROMA_API_TOKEN") or "YOUR-CHROMA-API-TOKEN"
CHROMA_TENANT = os.getenv("CHROMA_TENANT") or None
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE") or None

BUILD_TAG = 'beta'

client = ApifyClient(APIFY_API_TOKEN)

print("Starting Apify's Website Content Crawler")
actor_call = client.actor("apify/website-content-crawler").call(
    run_input={"maxCrawlPages": 5, "startUrls": [{"url": "https://docs.trychroma.com/"}]}
)
print("Actor website content crawler finished")
print(actor_call)

chroma_integration_inputs = {
    "chromaCollectionName": "apify",
    "chromaClientHost": CHROMA_HOST,
    "chromaClientSsl": True,
    "datasetId": actor_call["defaultDatasetId"],
    "datasetFields": ["text"],
    "dataUpdatesStrategy": "deltaUpdates",
    "dataUpdatesPrimaryDatasetFields": ["url"],
    "deleteExpiredObjects": True,
    "expiredObjectDeletionPeriodDays": 30,
    "embeddingsProvider": "OpenAI",
    "embeddingsApiKey": OPENAI_API_KEY,
    "embeddingsConfig": {"model": "text-embedding-3-small"},
    "performChunking": False,
    "chunkSize": 1000,
    "chunkOverlap": 0,
}

# Add optional ChromaDB parameters if they exist
if CHROMA_API_TOKEN:
    chroma_integration_inputs["chromaApiToken"] = CHROMA_API_TOKEN
if CHROMA_TENANT:
    chroma_integration_inputs["chromaTenant"] = CHROMA_TENANT
if CHROMA_DATABASE:
    chroma_integration_inputs["chromaDatabase"] = CHROMA_DATABASE

chroma_integration_inputs["build"] = BUILD_TAG

print("Starting Apify's ChromaDB Integration")
actor_call = client.actor("apify/chroma-integration").call(run_input=chroma_integration_inputs)
print("Apify's ChromaDB Integration has finished")
print(actor_call)

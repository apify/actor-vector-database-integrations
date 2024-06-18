# type: ignore
"""
Apify Qdrant integration
Use Apify's Qdrant integration to transfer website content data into a Qdrant database.

This example uses the Website Content Crawler to crawl the Qdrant documentation website and uploads the data to a Qdrant database.

pip install apify-client
"""

import os

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv("../../.env")

# Load environment variables from .env file or specify them here
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN") or "YOUR-APIFY-TOKEN"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY"
QDRANT_URL = os.getenv("QDRANT_URL") or "YOUR-QDRANT-URL"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or "YOUR-QDRANT-API-KEY"

client = ApifyClient(APIFY_API_TOKEN)

print("Starting Apify's Website Content Crawler")
actor_call = client.actor("apify/website-content-crawler").call(
    run_input={"maxCrawlPages": 10, "startUrls": [{"url": "https://qdrant.tech/documentation/"}]}
)
print("Actor website content crawler finished")
print(actor_call)

qdrant_integration_inputs = {
    "qdrantUrl": QDRANT_URL,
    "qdrantApiKey": QDRANT_API_KEY,
    "qdrantCollectionName": "apify",
    "qdrantAutoCreateCollection": True,
    "datasetId": actor_call["defaultDatasetId"],
    "datasetFields": ["text"],
    "enableDeltaUpdates": True,
    "deltaUpdatesPrimaryDatasetFields": ["url"],
    "expiredObjectDeletionPeriodDays": 30,
    "embeddingsProvider": "OpenAI",
    "embeddingsApiKey": OPENAI_API_KEY,
    "performChunking": False,
    "chunkSize": 1000,
    "chunkOverlap": 0,
}

print("Starting Apify's Qdrant Integration")
actor_call = client.actor("apify/qdrant-integration").call(run_input=qdrant_integration_inputs)
print("Apify's Pinecone Integration has finished")
print(actor_call)

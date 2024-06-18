# type: ignore
"""
Apify pinecone integration
Use Apify's Pinecone integration to transfer website content data into a Pinecone database.

This example uses the Website Content Crawler to crawl the Pinecone documentation website and transfer the data to a Pinecone database.

pip install apify-client
"""
import os

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv("../../.env")

# Load environment variables from .env file or specify them here (see .env.example)
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN") or "YOUR-APIFY-TOKEN"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "YOUR-PINECONE-API-KEY"

client = ApifyClient(APIFY_API_TOKEN)


print("Starting Apify's Website Content Crawler")
actor_call = client.actor("apify/website-content-crawler").call(
    run_input={"maxCrawlPages": 10, "startUrls": [{"url": "https://docs.pinecone.io/home"}, {"url": "https://docs.pinecone.io/integrations/apify"}]}
)
print("Actor website content crawler finished")
print(actor_call)

pinecone_integration_inputs = {
    "pineconeApiKey": PINECONE_API_KEY,
    "pineconeIndexName": "apify",
    "datasetFields": ["text"],
    "datasetId": actor_call["defaultDatasetId"],
    "enableDeltaUpdates": True,
    "deltaUpdatesPrimaryDatasetFields": ["url"],
    "expiredObjectDeletionPeriodDays": 30,
    "embeddingsApiKey": OPENAI_API_KEY,
    "embeddingsProvider": "OpenAI",
    "performChunking": True,
    "chunkSize": 1000,
    "chunkOverlap": 0,
}

print("Starting Apify's Pinecone Integration")
actor_call = client.actor("apify/pinecone-integration").call(run_input=pinecone_integration_inputs)
print("Apify's Pinecone Integration has finished")
print(actor_call)

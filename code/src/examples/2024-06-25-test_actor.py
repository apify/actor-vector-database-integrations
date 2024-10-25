import os

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

client = ApifyClient(token=os.getenv("APIFY_API_TOKEN"))

actor_name = "apify/opensearch-integration"

run_input = {
    "datasetId": "YOUR-DATASET-ID",
    "datasetFields": ["text"],
    "deltaUpdatesPrimaryDatasetFields": ["url"],
    "openSearchIndexName": "apify-index",
    "embeddingsApiKey": os.getenv("OPENAI_API_KEY"),
    "embeddingsConfig": {"model": "text-embedding-3-small"},
    "embeddingsProvider": "OpenAI",
    "enableDeltaUpdates": True,
    "openSearchUrl": os.getenv("OPENSEARCH_URL"),
    "awsAccessKeyId": os.getenv("AWS_ACCESS_KEY_ID"),
    "awsSecretAccessKey": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "awsRegion": "us-east-1",
}

result = client.actor(actor_name).call(run_input=run_input)
print(result)

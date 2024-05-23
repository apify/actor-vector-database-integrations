from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = ""
APIFY_API_TOKEN = ""
PINECONE_TOKEN = ""

pinecone_integration_inputs = {
    "index_name": "apify",
    "pinecone_token": PINECONE_TOKEN,
    "openai_token": OPENAI_API_KEY,
    "fields": ["text"],
    "perform_chunking": True,
    "chunk_size": 2048,
    "chunk_overlap": 0,
}

apify_client = ApifyClient(APIFY_API_TOKEN)

actor_call = apify_client.actor("apify/website-content-crawler").call(
    run_input={"maxCrawlPages": 1, "startUrls": [{"url": "https://docs.pinecone.io/home"}]}
)

print("Actor website content crawler finished", actor_call)

pinecone_integration_inputs["dataset_id"] = actor_call["defaultDatasetId"]
actor_call = apify_client.actor("jan.turon/pinecone-integration").call(run_input=pinecone_integration_inputs)
print("Apify's pinecone integration finished", actor_call)

import requests
import os
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.apify.com/v2/datasets"

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

print(APIFY_API_TOKEN)
r = requests.post(URL, json={"data": "my-data"}, params={"name": "my-dataset-name", "token": APIFY_API_TOKEN})
print(r.json())
print(r.url)


# https://api.apify.com/v2/datasets?name=my-dataset-name&token=apify_api_IoPOM26vW1hV4tqum7jVYsoFvm0UZt4iEOPH

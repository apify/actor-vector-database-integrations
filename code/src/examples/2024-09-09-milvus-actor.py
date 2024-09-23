# type: ignore
"""
Retrieval-Augmented Generation: Use Apify's Website Content Crawler to extract website content and store it in a Milvus database for question answering.

This tutorial demonstrates how to use the Website Content Crawler and Milvus/Zilliz Integration Actors from Apify to crawl website content
 and save it into a Milvus database. The stored data is then used to answer questions.

Expected output:
Question: What is Milvus database?
Answer: Milvus is a highly performant distributed vector database designed for managing and retrieving unstructured data using vector embeddings.
 It excels in scalable similarity searches, making it suitable for applications such as AI, machine learning, and semantic search.
 Unlike traditional databases that handle structured data with precise search operations, Milvus uses techniques like
 the Approximate Nearest Neighbor (ANN) algorithm to facilitate semantic similarity searches across diverse types of unstructured data,
 including images, audio, videos, and text ....
"""
import os

from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus.vectorstores import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv("../../.env")

# Load environment variables from .env file or specify them here
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN") or "YOUR-APIFY-TOKEN"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY"

MILVUS_COLLECTION_NAME = "apify"
MILVUS_URI = os.getenv("MILVUS_URI") or "YOUR-MILVUS-URI"
MILVUS_API_KEY = os.getenv("MILVUS_API_KEY") or "YOUR-MILVUS-API-KEY"

client = ApifyClient(APIFY_API_TOKEN)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Starting Apify's Website Content Crawler")
print("Crawling will take some time ... you can check the progress in the Apify console")

actor_call = client.actor(actor_id="apify/website-content-crawler").call(
    run_input={"maxCrawlPages": 10, "startUrls": [{"url": "https://milvus.io/"}, {"url": "https://zilliz.com/"}]}
)

print("Actor website content crawler has finished")
print(actor_call)

milvus_integration_inputs = {
    "milvusUri": MILVUS_URI,
    "milvusApiKey": MILVUS_API_KEY,
    "milvusCollectionName": MILVUS_COLLECTION_NAME,
    "datasetFields": ["text"],
    "datasetId": actor_call["defaultDatasetId"],
    "deltaUpdatesPrimaryDatasetFields": ["url"],
    "expiredObjectDeletionPeriodDays": 30,
    "embeddingsApiKey": OPENAI_API_KEY,
    "embeddingsProvider": "OpenAI",
}

print("Starting Apify's Milvus/Zilliz Integration")
actor_call = client.actor("apify/milvus-integration").call(run_input=milvus_integration_inputs)
print("Apify's Milvus/Zilliz Integration has finished")
print(actor_call)

print("Question answering using Milvus/Zilliz database")
vectorstore = Milvus(
    connection_args={"uri": MILVUS_URI, "token": MILVUS_API_KEY},
    embedding_function=embeddings,
    collection_name=MILVUS_COLLECTION_NAME,
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following pieces of retrieved context to answer the question. If you don't know the answer, "
    "just say that you don't know. \nQuestion: {question} \nContext: {context} \nAnswer:",
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

question = "What is Milvus database?"

print("Question:", question)
print("Answer:", rag_chain.invoke(question))

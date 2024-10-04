import enum

VCR_HEADERS_EXCLUDE = ["Authorization", "Api-Key"]

DAY_IN_SECONDS = 24 * 3600
BACKOFF_MAXTIME_SECONDS = 120


class SupportedVectorStores(str, enum.Enum):
    chroma = "chroma"
    milvus = "milvus"
    pgvector = "pgvector"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class SupportedEmbeddings(str, enum.Enum):
    openai = "OpenAI"
    cohere = "Cohere"
    fake = "Fake"

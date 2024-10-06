import enum

VCR_HEADERS_EXCLUDE = ["Authorization", "Api-Key"]

DAY_IN_SECONDS = 24 * 3600
BACKOFF_MAX_TIME_SECONDS = 120
BACKOFF_MAX_TIME_DELETE_SECONDS = 300  # 5 minutes (if many objects were added it takes time to search in the database)


class SupportedVectorStores(str, enum.Enum):
    chroma = "chroma"
    milvus = "milvus"
    opensearch = "opensearch"
    pgvector = "pgvector"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class SupportedEmbeddings(str, enum.Enum):
    openai = "OpenAI"
    cohere = "Cohere"
    fake = "Fake"

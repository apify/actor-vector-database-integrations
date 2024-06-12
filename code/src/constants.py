import enum

VCR_HEADERS_EXCLUDE = ["Authorization", "Api-Key"]


class SupportedVectorStores(str, enum.Enum):
    pinecone = "pinecone"
    chroma = "chroma"
    qdrant = "qdrant"


class SupportedEmbeddings(str, enum.Enum):
    openai = "OpenAI"
    cohere = "Cohere"

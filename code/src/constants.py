import enum

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"


class SupportedVectorStoresEn(str, enum.Enum):
    pinecone = "pinecone"
    chroma = "chroma"


class SupportedEmbeddingsEn(str, enum.Enum):
    open_ai_embeddings = "OpenAIEmbeddings"
    cohere_embeddings = "CohereEmbeddings"
    hugging_face_embeddings = "HuggingFaceEmbeddings"

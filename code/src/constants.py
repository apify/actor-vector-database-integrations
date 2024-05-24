import enum

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"


class SupportedVectorStores(str, enum.Enum):
    pinecone = "pinecone"
    chroma = "chroma"


class SupportedEmbeddings(str, enum.Enum):
    open_ai_embeddings = "OpenAIEmbeddings"
    cohere_embeddings = "CohereEmbeddings"
    hugging_face_embeddings = "HuggingFaceEmbeddings"

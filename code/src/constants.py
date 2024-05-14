import enum

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"


class SupportedVectorStoresEn(str, enum.Enum):
    pinecone = "pinecone"
    chroma = "chroma"

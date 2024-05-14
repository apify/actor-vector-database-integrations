import enum


class SupportedVectorStoresEn(str, enum.Enum):
    pinecone = "pinecone"
    chroma = "chroma"

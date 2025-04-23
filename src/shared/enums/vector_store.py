from enum import Enum


class VectorStoreEnum(str, Enum):
    chroma = "chroma"
    faiss = "faiss"

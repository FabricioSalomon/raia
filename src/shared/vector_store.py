from typing import Dict, Optional

from langchain.embeddings.base import Embeddings

from src.infrastructure.vector_store.base import BaseVectorStore
from src.infrastructure.vector_store.chroma import ChromaVectorStore
from src.infrastructure.vector_store.faiss import FaissVectorStore
from src.shared.enums.vector_store import VectorStoreEnum


class VectorStoreFactory:
    @staticmethod
    def get_vector_store(
        embedding: Embeddings,
        store_type: Optional[VectorStoreEnum] = VectorStoreEnum.faiss,
    ) -> BaseVectorStore:
        vector_store_map: Dict[VectorStoreEnum, BaseVectorStore] = {
            VectorStoreEnum.chroma: ChromaVectorStore(
                embedding=embedding,
                collection_name="ingestion",
            ),
            VectorStoreEnum.faiss: FaissVectorStore(embedding=embedding),
        }

        if store_type is None or store_type not in vector_store_map:
            raise ValueError(f"Unsupported vector store: {store_type}")

        return vector_store_map[store_type]

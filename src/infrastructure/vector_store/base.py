from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.shared.enums.vector_store import VectorStoreEnum


class BaseVectorStore(ABC):
    store_type: VectorStoreEnum = VectorStoreEnum.faiss

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        pass

    @abstractmethod
    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = 0.7,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass

    @abstractmethod
    def save_local(self, path: Optional[str]) -> None:
        pass

    @abstractmethod
    def load_local(self, path: str) -> None:
        pass

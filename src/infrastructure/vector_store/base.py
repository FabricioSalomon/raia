from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
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

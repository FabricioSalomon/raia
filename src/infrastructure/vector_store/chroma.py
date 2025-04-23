from typing import List, Optional

from langchain.embeddings.base import Embeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

from src.infrastructure.vector_store.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str, embedding: Embeddings):
        self.collection_name = collection_name
        self.embedding = embedding
        self.store = Chroma(
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )
        self.documents = []

    def add_documents(self, documents: List[Document]) -> None:
        self.store.add_documents(documents)
        self.documents.extend(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.store.similarity_search(query, k=k)

    def delete(self, ids: List[str]) -> None:
        self.store.delete(ids)

    def save_local(self, path: Optional[str] = None) -> None:
        if not path:
            raise ValueError("Path must be provided to save the Chroma vector store.")

        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=path,
        )

        if self.documents:
            self.store.add_documents(self.documents)

    def load_local(self, path: str) -> None:
        self.store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=path,
        )

import csv
import os
import tempfile
from typing import Callable, Dict, List

from fastapi import UploadFile
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from PyPDF2 import PdfReader

from src.api.schemas.ingestion import VectorStoreEnum
from src.infrastructure.vector_store.chroma import ChromaVectorStore
from src.infrastructure.vector_store.faiss import FaissVectorStore


class IngestionService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.splitter = SemanticChunker(self.embeddings)

        # Map file extensions to handler methods
        self._file_handlers: Dict[str, Callable[[bytes], str]] = {
            ".txt": self._handle_txt,
            ".csv": self._handle_csv,
            ".pdf": self._handle_pdf,
        }

    async def process(
        self, files: List[UploadFile], vector_store: VectorStoreEnum
    ) -> None:

        store = self._get_vector_store(vector_store)

        for file in files:
            contents = await file.read()
            extension = os.path.splitext(file.filename or "")[1].lower()
            handler = self._file_handlers.get(extension)

            if not handler:
                raise ValueError(f"Unsupported file type: {extension}")

            text = handler(contents)
            chunks = self.splitter.create_documents([text])
            store.add_documents(chunks)

        if vector_store == VectorStoreEnum.chroma:
            store.save_local("./chroma_index")

        elif vector_store == VectorStoreEnum.faiss:
            store.save_local("./faiss_index")

        else:
            raise ValueError("Unsupported vector store selected.")

    def _get_vector_store(self, store_type: VectorStoreEnum):
        if store_type == VectorStoreEnum.chroma:
            return ChromaVectorStore(
                collection_name="ingestion", embedding=self.embeddings
            )
        elif store_type == VectorStoreEnum.faiss:
            return FaissVectorStore(embedding=self.embeddings)
        raise ValueError(f"Unsupported vector store: {store_type}")

    def _handle_txt(self, content: bytes) -> str:
        return content.decode("utf-8")

    def _handle_csv(self, content: bytes) -> str:
        decoded = content.decode("utf-8").splitlines()
        reader = csv.reader(decoded)
        rows = [" ".join(row) for row in reader]
        return "\n".join(rows)

    def _handle_pdf(self, content: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            reader = PdfReader(tmp_path)
            text = "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        finally:
            os.remove(tmp_path)

        return text

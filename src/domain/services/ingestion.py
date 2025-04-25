import os
import tempfile
from typing import Callable, Dict, List, Optional, Union, cast

import pandas as pd
from fastapi import UploadFile
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from PyPDF2 import PdfReader

from src.api.schemas.ingestion import VectorStoreEnum
from src.infrastructure.vector_store.chroma import ChromaVectorStore
from src.infrastructure.vector_store.faiss import FaissVectorStore
from src.shared.csv_formatter import CSV

FileHandler = Dict[str, Callable[[bytes], Union[str, pd.DataFrame]]]


class IngestionService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
        )

        self._file_handlers: FileHandler = {
            ".txt": self._handle_txt,
            ".csv": self._handle_csv,
            ".pdf": self._handle_pdf,
        }

    async def process(
        self,
        files: List[UploadFile],
        column_to_ingest: Optional[str] = "text",
        vector_store: Optional[VectorStoreEnum] = VectorStoreEnum.faiss,
    ) -> None:
        if not vector_store:
            raise ValueError(f"Empty vector store.")
        store = self._get_vector_store(vector_store)

        for file in files:
            contents = await file.read()
            extension = os.path.splitext(file.filename or "")[1].lower()
            handler = self._file_handlers.get(extension)

            if not handler:
                raise ValueError(f"Unsupported file type: {extension}")

            text = handler(contents)
            chunks: List[Document] = []
            if extension == ".csv" and column_to_ingest:
                dataframe = cast(pd.DataFrame, text)
                for index, row in dataframe.head(10).iterrows():
                    column_data = row[column_to_ingest]
                    chunks.append(
                        Document(
                            page_content=column_data,
                            metadata={
                                "question_id": row["id"],
                                "source": file.filename,
                                "metadata": {
                                    "file_name": file.filename,
                                    "file_type": extension,
                                    "file_size": len(contents),
                                },
                            },
                        )
                    )
            else:
                text = cast(str, text)
                chunks = self.splitter.create_documents(
                    [text],
                    [
                        {
                            "source": file.filename,
                            "metadata": {
                                "file_name": file.filename,
                                "file_type": extension,
                                "file_size": len(contents),
                            },
                        }
                    ],
                )
            store.add_documents(chunks)

        store.save_local(f"./{vector_store.value}_index")

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

    def _handle_csv(self, content: bytes) -> pd.DataFrame:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        if tmp_path is None:
            raise ValueError("Temporary file path is None")
        try:
            service = CSV(tmp_path)

            data = service.read()

        finally:
            os.remove(tmp_path)
        return data

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

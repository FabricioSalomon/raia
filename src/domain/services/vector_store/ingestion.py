import os
import tempfile
from typing import Callable, Dict, List, Optional, Union, cast

import pandas as pd
from fastapi import UploadFile
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from PyPDF2 import PdfReader

from src.shared.csv_formatter import CSV
from src.shared.enums.vector_store import VectorStoreEnum
from src.shared.vector_store import VectorStoreFactory

FileHandler = Dict[str, Callable[[bytes], Union[str, pd.DataFrame]]]


class IngestionService:
    def __init__(
        self,
        embedding: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY", ""),
        ),
        vector_store: Optional[VectorStoreEnum] = VectorStoreEnum.faiss,
    ):
        self.embeddings = embedding
        self.splitter = SemanticChunker(
            embeddings=embedding,
        )
        self.store = VectorStoreFactory.get_vector_store(
            embedding=embedding,
            store_type=vector_store,
        )

        self._file_handlers: FileHandler = {
            ".txt": self._handle_txt,
            ".csv": self._handle_csv,
            ".pdf": self._handle_pdf,
        }

    async def invoke(
        self,
        files: List[UploadFile],
        column_to_embed: str = "text",
        vector_store_path: Optional[str] = "faiss_index",
    ) -> None:

        for file in files:
            contents = await file.read()
            extension = os.path.splitext(file.filename or "")[1].lower()
            handler = self._file_handlers.get(extension)

            if not handler:
                raise ValueError(f"Unsupported file type: {extension}")

            self.file_metadata = {
                "file_type": extension,
                "file_name": file.filename,
                "file_size": len(contents),
            }
            text = handler(contents)
            chunks: List[Document] = []
            if extension == ".csv" and column_to_embed:
                dataframe = cast(pd.DataFrame, text)
                for index, row in dataframe.iterrows():
                    self._ingest_csv_column(
                        row=row,
                        column_to_embed=column_to_embed,
                    )
            else:
                text = cast(str, text)
                chunks = self.splitter.create_documents(
                    texts=[text],
                    metadatas=[
                        {
                            "file_metadata": self.file_metadata,
                        },
                    ],
                )
                self.store.add_documents(chunks)

        self.store.save_local(
            f"./assets/{self.store.store_type.value}/{vector_store_path}"
        )

    def _ingest_csv_column(
        self,
        row: pd.Series,
        column_to_embed: str,
    ):
        column_data = row[column_to_embed]
        chunk = [
            Document(
                page_content=column_data,
                metadata={
                    "question_id": row["id"],
                    "subject": row["subject"],
                    "university": row["source"].split()[0],
                    "file_metadata": self.file_metadata,
                },
            )
        ]
        self.store.add_documents(chunk)

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

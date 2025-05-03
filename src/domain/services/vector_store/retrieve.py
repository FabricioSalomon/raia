import os
import tempfile
from typing import Callable, Dict, Optional, Union

import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from PyPDF2 import PdfReader

from src.infrastructure.vector_store.chroma import ChromaVectorStore
from src.infrastructure.vector_store.faiss import FaissVectorStore
from src.shared.csv_formatter import CSV
from src.shared.enums.subjects import SubjectLiteral
from src.shared.enums.universities import UniversityLiteral
from src.shared.enums.vector_store import VectorStoreEnum
from src.shared.vector_store import VectorStoreFactory

FileHandler = Dict[str, Callable[[bytes], Union[str, pd.DataFrame]]]


class RetrieveService:
    def __init__(
        self,
        embedding: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY", ""),
        ),
        vector_store: Optional[VectorStoreEnum] = VectorStoreEnum.faiss,
    ):
        self.store = VectorStoreFactory.get_vector_store(
            embedding=embedding,
            store_type=vector_store,
        )
        self.embeddings = embedding

    def invoke(
        self,
        message: str,
        subject: Optional[SubjectLiteral],
        university: Optional[UniversityLiteral],
        vector_store_name: Optional[str] = "faiss_index",
    ):
        if not vector_store_name:
            raise ValueError(f"Empty vector store name.")
        faiss_store = FaissVectorStore()
        faiss_store.load_local(path=vector_store_name)
        response = faiss_store.similarity_search_with_relevance_scores(
            message,
            # filter={
            #     "university": university if university else None,
            #     "subject": subject if subject else None,
            # },
        )

        return response

    def _get_vector_store(
        self, store_type: Optional[VectorStoreEnum] = VectorStoreEnum.faiss
    ):
        vector_store_map = {
            VectorStoreEnum.chroma: ChromaVectorStore(
                collection_name="ingestion", embedding=self.embeddings
            ),
            VectorStoreEnum.faiss: FaissVectorStore(embedding=self.embeddings),
        }

        if not store_type or store_type not in vector_store_map:
            raise ValueError(f"Unsupported vector store: {store_type}")

        return vector_store_map[store_type]

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

from typing import Optional

from pydantic import BaseModel, Field

from src.shared.enums.vector_store import VectorStoreEnum


class IngestionMetadata(BaseModel):
    column_to_ingest: Optional[str] = Field(
        ..., description="CSV Column reference for ingestion"
    )
    vector_store: Optional[VectorStoreEnum] = Field(
        ..., description="Vector store to use"
    )

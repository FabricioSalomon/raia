from pydantic import BaseModel, Field

from src.shared.enums.vector_store import VectorStoreEnum


class IngestionMetadata(BaseModel):
    vector_store: VectorStoreEnum = Field(..., description="Vector store to use")

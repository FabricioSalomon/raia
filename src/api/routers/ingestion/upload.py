import json
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api.schemas.ingestion import IngestionMetadata
from src.domain.services.ingestion import IngestionService
from src.exceptions.api.bad_request import BadRequestException
from src.exceptions.api.internal_server import InternalServerErrorException
from src.shared.enums.vector_store import VectorStoreEnum

upload_router = APIRouter()


@upload_router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        parsed_metadata = IngestionMetadata(
            vector_store=VectorStoreEnum.faiss,
            column_to_ingest="text",
        )
        if metadata:
            metadata_dict = json.loads(metadata)
            parsed_metadata = IngestionMetadata(**metadata_dict)
        ingestion_service = IngestionService()
        await ingestion_service.process(
            files=files,
            vector_store=parsed_metadata.vector_store,
            column_to_ingest=parsed_metadata.column_to_ingest,
        )

        return JSONResponse(
            content={"message": "Documents processed and ingested successfully"},
            status_code=200,
        )

    except (json.JSONDecodeError, ValidationError) as e:
        raise BadRequestException(
            "[Router - Ingestion - Upload]",
            {
                "message": f"Invalid metadata: {str(e)}",
            },
        )
    except Exception as e:
        raise InternalServerErrorException(
            "[Router - Ingestion - Upload]",
            {
                "message": f"Ingestion failed: {str(e)}",
            },
        )

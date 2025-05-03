import json
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.domain.services.vector_store.ingestion import IngestionService
from src.exceptions.api.bad_request import BadRequestException
from src.exceptions.api.internal_server import InternalServerErrorException
from src.shared.enums.vector_store import VectorStoreEnum

upload_router = APIRouter()


@upload_router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    column_to_embed: str = Form(...),
    vector_store_path: Optional[str] = Form(None),
    vector_store: Optional[VectorStoreEnum] = Form(None),
):
    try:

        ingestion_service = IngestionService(
            vector_store=vector_store,
        )
        await ingestion_service.invoke(
            files=files,
            column_to_embed=column_to_embed,
            vector_store_path=vector_store_path,
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

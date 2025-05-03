import json
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.domain.services.vector_store.retrieve import RetrieveService
from src.exceptions.api.bad_request import BadRequestException
from src.exceptions.api.internal_server import InternalServerErrorException
from src.shared.enums.subjects import SubjectLiteral
from src.shared.enums.universities import UniversityLiteral
from src.shared.enums.vector_store import VectorStoreEnum

retrieve_router = APIRouter()


@retrieve_router.post("/retrieve")
async def retrieve_questions(
    message: str = Form(None),
    subject: Optional[SubjectLiteral] = Form(None),
    university: Optional[UniversityLiteral] = Form(None),
    vector_store_name: Optional[str] = Form("faiss_index"),
    vector_store: Optional[VectorStoreEnum] = Form(VectorStoreEnum.faiss),
):
    try:
        service = RetrieveService(vector_store=vector_store)
        response = service.invoke(
            message,
            subject=subject,
            university=university,
            vector_store_name=vector_store_name,
        )

        return JSONResponse(
            content={
                "message": "Documents analyzed and retrieved successfully",
                "data": json.dumps(f"{response}"),
            },
            status_code=200,
        )

    except (json.JSONDecodeError, ValidationError) as e:
        raise BadRequestException(
            "[Router - Ingestion - Retrieve]",
            {
                "message": f"Invalid metadata: {str(e)}",
            },
        )
    except Exception as e:

        raise InternalServerErrorException(
            "[Router - Ingestion - Retrieve]",
            {
                "message": f"Ingestion failed: {str(e)}",
            },
        )

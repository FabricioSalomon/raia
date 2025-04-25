import json
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.exceptions.api.bad_request import BadRequestException
from src.exceptions.api.internal_server import InternalServerErrorException
from src.infrastructure.vector_store.faiss import FaissVectorStore
from src.shared.enums.subjects import SubjectLiteral
from src.shared.enums.universities import UniversityLiteral
from src.shared.llm import LLM

retrieve_router = APIRouter()


@retrieve_router.post("/retrieve")
async def retrieve_questions(
    message: str = Form(None),
    university: Optional[UniversityLiteral] = Form(None),
    subject: Optional[SubjectLiteral] = Form(None),
):
    try:
        LLM(
            model="openai",
            model_name="gpt-4o",
        )
        faiss_store = FaissVectorStore()
        faiss_store.load_local("faiss_index")

        response = faiss_store.similarity_search(message)

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

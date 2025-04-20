from fastapi import APIRouter, Depends, HTTPException

# from src.dependencies import get_token_header
from src.controllers.questions.generate.index import GenerateController

controller = GenerateController()

questions_router = APIRouter(
    prefix="/questions",
    tags=["questions"],
    # dependencies=[Depends(get_token_header)],
)

questions_router.post("/generate")(controller.invoke)

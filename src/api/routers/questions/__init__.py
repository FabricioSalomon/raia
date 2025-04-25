from fastapi import APIRouter

# from src.infrastructure.db.session import get_db
from .generate import generate_questions_router

questions_router = APIRouter(
    prefix="/questions",
    tags=["questions"],
    # dependencies=[Depends(get_db)],
)

questions_router.include_router(generate_questions_router)

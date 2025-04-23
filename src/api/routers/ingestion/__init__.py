from fastapi import APIRouter, Depends

from src.infrastructure.db.session import get_db

from .upload import upload_router

ingestion_router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
    dependencies=[Depends(get_db)],
)

ingestion_router.include_router(upload_router)

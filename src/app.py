from fastapi import FastAPI

from src.api.routers.ingestion import ingestion_router
from src.api.routers.questions import questions_router
from src.infrastructure.logging_config import setup_logging

setup_logging()

app = FastAPI()
app.include_router(questions_router)
app.include_router(ingestion_router)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
    }

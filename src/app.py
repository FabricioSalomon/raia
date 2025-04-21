from fastapi import FastAPI

from infrastructure.logging_config import setup_logging
from src.api.routers.questions import questions_router

setup_logging()

app = FastAPI()
app.include_router(questions_router)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
    }

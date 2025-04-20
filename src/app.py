from fastapi import FastAPI
from .routers.questions.index import questions_router

app = FastAPI()


app.include_router(questions_router)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
    }

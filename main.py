import uvicorn
from fastapi import FastAPI

from src.app import app

mainApp = FastAPI()
mainApp.mount("/api", app)


if __name__ == "__main__":
    uvicorn.run(
        mainApp,
        host="0.0.0.0",
        log_level="info",
    )

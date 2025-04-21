import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from infrastructure.db.config import Settings

settings = Settings(
    DB_USER=os.getenv("DB_USER", ""),
    DB_PASSWORD=os.getenv("DB_PASSWORD", ""),
    DB_NAME=os.getenv("DB_NAME", "database"),
    DB_HOST=os.getenv("DB_HOST", "localhost"),
    DB_PORT=os.getenv("DB_PORT", "5432"),
)

engine = create_engine(
    url=settings.sqlalchemy_database_uri,
    echo=True,
)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

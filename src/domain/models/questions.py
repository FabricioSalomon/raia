from sqlalchemy import UUID, Column

from src.infrastructure.db.session import Base


class Question(Base):
    __tablename__ = "questions"

    id = Column(
        UUID,
        primary_key=True,
        index=True,
    )

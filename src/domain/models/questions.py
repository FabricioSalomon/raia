from sqlalchemy import UUID, Column

from infrastructure.db.session import Base


class Question(Base):
    __tablename__ = "questions"

    id = Column(
        UUID,
        primary_key=True,
        index=True,
    )

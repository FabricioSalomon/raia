from typing import List

from pydantic import BaseModel, Field


class ResponseFormatter(BaseModel):
    """Always use this class to structure your response to the user."""

    subject: List[str] = Field(
        description="The schools subjects the exam question is most related. It can be more than one option. E.g.: history, math, grammar..."
    )
    question_number: int = Field(description="The question number in the exam")
    question: int = Field(description="The question text in the exam")
    answer: str = Field(description="The question answer based on the given options")


class ResponseList(BaseModel):
    list: List[ResponseFormatter]

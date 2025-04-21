from enum import Enum
from typing import Literal

from pydantic import BaseModel


class Subjects(BaseModel):
    geography: bool
    history: bool
    biology: bool
    math: bool
    other: bool


class SubjectEnum(Enum):
    MATH = "math"
    HISTORY = "history"
    BIOLOGY = "biology"
    GEOGRAPHY = "geography"


SubjectLiteral = Literal[
    "math",
    "biology",
    "history",
    "geography",
]

from enum import Enum
from typing import Literal, NotRequired, Optional, Required, Union
from typing_extensions import TypedDict
from pydantic import UUID4


class SubjectEnum(Enum):
    GEOGRAPHY = "geography"
    HISTORY = "history"
    MATH = "math"


SubjectLiteral = Literal[
    "geography",
    "history",
    "math",
]


class UniversityEnum(Enum):
    USP = "USP"


UniversityLiteral = Literal["USP"]


class QuestionResponse(TypedDict):
    id: Required[UUID4]
    number: Required[int]
    content: Required[str]
    image: NotRequired[Optional[str]]


class Payload(TypedDict):
    user_message: Required[str]
    include_images: NotRequired[Optional[bool]]
    subject: Required[Union[SubjectEnum, SubjectLiteral]]
    university: NotRequired[Optional[Union[UniversityEnum, UniversityLiteral]]]

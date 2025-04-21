from enum import Enum
from typing import Literal, NotRequired, Optional, Required, Union
from typing_extensions import TypedDict
from pydantic import UUID4

from shared.enums.subjects import SubjectEnum, SubjectLiteral
from shared.enums.universities import UniversityEnum, UniversityLiteral


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

from enum import Enum
from typing import Literal


class UniversityEnum(Enum):
    USP = "USP"
    UFSCar = "UFSCar"


UniversityLiteral = Literal[
    "USP",
    "UFSCar",
]

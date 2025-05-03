from typing import Optional

from .base import APIException


class InternalServerErrorException(APIException):
    def __init__(
        self,
        resource: str,
        metadata: Optional[dict[str, str]] = None,
    ):
        super().__init__(500, f"Internal Server Error: {resource}", metadata)

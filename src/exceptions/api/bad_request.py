from typing import Optional

from .base import APIException


class BadRequestException(APIException):
    def __init__(
        self,
        resource: str,
        metadata: Optional[dict[str, str]] = None,
    ):
        super().__init__(400, f"Bad Request: {resource}", metadata)

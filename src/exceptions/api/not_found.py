from typing import Optional

from .base import APIException


class NotFoundException(APIException):
    def __init__(
        self,
        resource: str,
        metadata: Optional[dict[str, str]] = None,
    ):
        super().__init__(404, f"{resource} not found", metadata)

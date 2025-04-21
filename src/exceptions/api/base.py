from typing import Optional

from fastapi import HTTPException


class APIException(HTTPException):
    def __init__(
        self,
        status_code: int,
        message: str,
        metadata: Optional[dict] = None,
    ):
        self.message = message
        self.metadata = metadata or {}
        super().__init__(
            status_code=status_code,
            detail={
                "message": self.message,
                "metadata": self.metadata,
            },
        )

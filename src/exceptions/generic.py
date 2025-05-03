from typing import Optional


class GenericException(Exception):
    def __init__(
        self,
        message: str,
        metadata: Optional[dict] = None,
    ):
        self.message = message
        self.metadata = metadata or {}
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} | Metadata: {self.metadata}"

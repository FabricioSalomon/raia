from typing import List
import uuid
from .base import GenerateInterface
from .model import Payload, QuestionResponse


class GenerateController(GenerateInterface):
    def invoke(self, payload: Payload) -> List[QuestionResponse]:
        user_message = payload.get("user_message")

        return [
            {
                "number": 1,
                "content": user_message,
                "id": uuid.uuid4(),
            },
            {
                "number": 1,
                "image": user_message,
                "content": user_message,
                "id": uuid.uuid4(),
            },
        ]

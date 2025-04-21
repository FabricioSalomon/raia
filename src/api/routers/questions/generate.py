import uuid
from typing import List

from fastapi import APIRouter

from api.schemas.questions import Payload, QuestionResponse

generate_questions_router = APIRouter()


@generate_questions_router.post("/generate", response_model=List[QuestionResponse])
def generate(payload: Payload) -> List[QuestionResponse]:
    user_message = payload.get("user_message")
    # service = UserService(UserRepository(db))
    # user = service.get_user_by_id(user_id)
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

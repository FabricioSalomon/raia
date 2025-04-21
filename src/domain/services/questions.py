from src.exceptions.api.not_found import NotFoundException


class QuestionService:
    def __init__(self, question_repository: QuestionRepository):
        self.question_repository = question_repository

    def generate(self):
        user = self.question_repository.get_by_id()
        if not user:
            raise NotFoundException(
                "Question",
                {
                    "question_id": f"{''}",
                },
            )
        return user

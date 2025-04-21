import getpass
import os
from typing import List, Literal, Optional, cast

import dotenv
from langchain.chat_models import init_chat_model

from src.shared.enums.subjects import Subject, SubjectEnum

dotenv.load_dotenv()


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def gpt_invoke(self, text: str) -> Optional[str]:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

        llm = init_chat_model(
            model=self.model_name,
            model_provider="openai",
        )

        llm_structured = llm.with_structured_output(Subject)
        response_structured = cast(Subject, llm_structured.invoke(text))
        if response_structured.enum == SubjectEnum.OTHER:
            messages = [
                {
                    "role": "ai",
                    "content": """
                        Considere as possíveis disciplinas de ensino médio para assuntos de vestibulares como: USP, UFSCar e ENEM.
                        Qual a disciplina da questão? 
                        
                        !!!!! Responda com uma palavra apenas !!!!!!
                    """,
                },
                {
                    "role": "human",
                    "content": text,
                },
            ]
            response = llm.invoke(messages)
            if response and response.content and isinstance(response.content, str):
                response_structured = cast(
                    Subject, llm_structured.invoke(response.content)
                )
        return response_structured.enum.value.lower()

    def gemini_invoke(self, text: str) -> str:
        return f"Processed by GEMINI: {text}"

    def claude_invoke(self, text: str) -> str:
        return f"Processed by CLAUDE: {text}"


class GenerateData(LLM):
    def __init__(self, model: Literal["GPT", "GEMINI", "CLAUDE"], model_name: str):
        self.model = model
        super().__init__(model_name)

    def invoke(self, text: str, alternatives: List[str]) -> str:
        hash_map = {
            "GPT": self.gpt_invoke,
            "GEMINI": self.gemini_invoke,
            "CLAUDE": self.claude_invoke,
        }
        if self.model not in hash_map:
            raise ValueError(f"Unsupported model: {self.model}")

        for alternative_text in alternatives:
            text = text + f", {alternative_text}"
        data = hash_map[self.model](text)
        print(f"GenerateData processed: {data} using {self.model}: {self.model_name}")
        return data

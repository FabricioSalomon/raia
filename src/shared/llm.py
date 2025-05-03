from typing import Literal, Optional

import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

dotenv.load_dotenv()

Models = Literal["openai", "gemini", "claude"]


class LLM:
    def __init__(
        self,
        model: Models,
        model_name: str,
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = 100,
    ):
        self.model: Models = model
        models = ["openai", "gemini", "claude"]
        if self.model not in models:
            raise ValueError(f"Unsupported model: {self.model}")

        self.llm: BaseChatModel = init_chat_model(
            model=model_name,
            model_provider=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

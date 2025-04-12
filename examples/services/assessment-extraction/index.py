import getpass
import os
from typing import List, cast

import dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.language_models import (
    BaseChatModel,
)
from models.llm_provider import LLMProvider, provider_mapper
from models.llm_response_formatter import ResponseList
from models.payload import Payload

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


class AssessmentExtraction:
    def __init__(self, model_provider: LLMProvider):
        self._model = model_provider["model"]
        self._model_provider = model_provider["provider"]
        self._llm = init_chat_model(
            model=self._model,
            model_provider=provider_mapper[self._model_provider],
        )

    def invoke(self, payload: Payload):
        origin = payload["file_origin"]
        if origin == "external":
            pass
        path = payload["file_path"]
        loader = PyPDFLoader(file_path=path)
        file = loader.load()

        structured = cast(
            BaseChatModel, self._llm.with_structured_output(ResponseList)  # type: ignore
        )

        # FewShot necessary
        messages = [
            {
                "role": "system",
                "content": """\n
                    Você é um especialista em vestibular da USP. Extraia as informações de cada questão.\n
                    Localize cada uma das questões, uma por uma, veja o número da questão, o conteúdo dela baseado\n
                    nas possíveis matérias de ensino médio e qual a possível resposta dentre as alternativas.\n
                    São 90 questões no total, portanto, serão 90 análises.\n
                """,
            }
        ]
        for doc in file:
            messages.append(
                {
                    "role": "assistant",
                    "content": doc.page_content,
                }
            )

        response = structured.invoke(messages)
        breakpoint()


extractor = AssessmentExtraction(
    model_provider={
        "model": "gpt-4o-mini",
        "provider": "GPT",
    }
)

extractor.invoke(
    {
        "file_origin": "local",
        "file_path": "https://www.fuvest.br/wp-content/uploads/fuvest2025_primeira_fase_prova_V1.pdf",
    }
)

from enum import Enum
from typing import Dict, Literal, TypedDict, Union


class LLMProviderEnum(str, Enum):
    GPT = "GPT"
    GEMINI = "GEMINI"


ProviderLiteral = Literal["GPT", "GEMINI"]
Provider = Union[ProviderLiteral, LLMProviderEnum]

provider_mapper: Dict[Provider, str] = {
    LLMProviderEnum.GPT.value: "openai",
    LLMProviderEnum.GEMINI.value: "gemini",
}


class LLMProvider(TypedDict):
    model: str
    provider: Provider

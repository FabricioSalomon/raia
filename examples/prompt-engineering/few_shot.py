import getpass
import os
from typing import List, NotRequired, Optional, TypedDict, cast

import dotenv
from datasets import load_dataset
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


class SourceData(TypedDict):
    id: str
    IU: bool
    exam: str
    label: str
    ledor: bool
    question: str
    alternatives: List[str]
    figures: NotRequired[List[str]]
    description: NotRequired[List[str]]


class FewShotData(TypedDict):
    question_number: str
    question: str
    image_description: Optional[str]
    alternatives: str
    answer: str


dotenv.load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
)


response = llm.invoke(
    """\
           Crie uma questão de ENEM seguindo o seguinte padrão:\
            Questão 1) Qual a capital do goiás?\
               a) Bahia\
               b) Goiânia\
               c) Curitiba\
               d) São Paulo\
               e) Natal\
        Considere um nível de dificuldade alta, pode ser de qualquer matéria, mas invente algo coerente com o nível ensino médio!\
        Não gere a resposta dessa questão, apenas a pergunta.
"""
)


ds = load_dataset("maritaca-ai/enem", "2024", split="train")
data_typed = cast(List[SourceData], ds)
examples: List[FewShotData] = []
for line in data_typed:
    example: FewShotData = {
        "answer": line.get("label"),
        "question": line.get("question"),
        "question_number": line.get("id"),
        "alternatives": f'{line.get("alternatives")}',
        "image_description": line.get("image_description"),
    }
    examples.append(example)

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question_number}: {question}\n{image_description}\n{alternatives}"),
        ("ai", "{answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é o melhor aluno da melhor universidade do mundo."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | llm

final_response = chain.invoke({"input": response.content})

print(
    f"""
        "question": {response.content},\\
        "llm_response": {final_response.content},\\
    """
)


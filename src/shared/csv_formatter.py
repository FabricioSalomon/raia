import re
from collections.abc import Hashable
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

import pandas as pd
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.exceptions.generic import GenericException
from src.shared.enums.subjects import Subject, SubjectEnum

from .llm import LLM, Models

T = TypeVar("T", str, int, float, bool)


class QuestionRow(TypedDict):
    id: int
    text: str
    subject: str
    category: str
    subcategory: str
    source: str
    difficulty: str
    parameter_A: str
    parameter_B: str
    parameter_C: str
    answer_text: str
    correct_answer: str
    A: str
    B: str
    C: str
    D: str
    E: str
    is_discursive: bool


class BaseEnum(BaseModel):
    enum: Enum


AvailableTypes = Union[int, str, bool, float]
QuestionRowTypeHints = Dict[str, Type[AvailableTypes]]


class CSV:
    def __init__(self, file_path: str) -> None:
        self.column: str = "text"
        self.file_path: str = file_path
        self.data: pd.DataFrame = self._read_csv_with_header()
        self.model: Models = "openai"
        self.model_name = "gpt-4o"
        self.llm: Optional[BaseChatModel] = None

    def read(self) -> pd.DataFrame:
        return self.data

    def format(self, column: str, model: Models, model_name: str):
        if self.data is None:
            raise GenericException("The file is empty.", {"file_path": self.file_path})
        self.model = model
        self.model_name = model_name
        for index, row in self.data.iterrows():
            # text_value: bool = row.get("is_discursive", False)
            # if text_value:
            #     continue
            fill_value: Optional[AvailableTypes] = row[column]
            filled = self._fill_missing(
                current_row=row,
                column=column,
                current_index=index,
                fill_value=fill_value,
            )
            if not filled:
                continue
            print(f"Filled with: [{index}] = {filled}")
        self.data.to_csv("../../assets/info.csv", index=False)
        return self.data

    def _fill_missing(
        self,
        column: str,
        current_row: pd.Series,
        current_index: Hashable,
        fill_value: Optional[AvailableTypes],
    ) -> Optional[AvailableTypes]:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        type_hints: QuestionRowTypeHints = get_type_hints(QuestionRow)
        column_type = type_hints.get(column)

        if not column_type:
            raise ValueError(f"Column '{column}' not found in QuestionRow annotations.")

        if not self._is_empty(row_value=fill_value):
            return

        fill_value = cast(AvailableTypes, fill_value)
        self.column = column
        if self.column == "subject" or self.column == "source":
            fill_value = self._generate_data_using_text_column(fill_value, current_row)

        if self._is_wrong_type_data(value=fill_value, type=column_type):
            raise TypeError(
                f"Expected {column_type} for column '{column}', but got {type(fill_value)}."
            )
        self.data.at[current_index, column] = fill_value
        return fill_value

    def _generate_data_using_text_column(
        self, fill_value: AvailableTypes, current_row: pd.Series
    ):
        text_value = current_row.get("text", "")
        if text_value.startswith("("):
            words = self._extract_words_in_parens(text_value)
            source_from_text = words[0] + f" {words[1]}"
            subject_from_text = words[-1]
            if source_from_text and self.column == "source":
                fill_value = source_from_text.upper()
            if subject_from_text and self.column == "subject":
                fill_value = subject_from_text.lower()
        else:

            if self.column == "source":
                fill_value = "AI"
            else:
                self.llm = LLM(
                    model=self.model,
                    model_name=self.model_name,
                    temperature=0.0,
                    max_tokens=100,
                ).llm
                alternatives = [
                    current_row.get("A", ""),
                    current_row.get("B", ""),
                    current_row.get("C", ""),
                    current_row.get("D", ""),
                    current_row.get("E", ""),
                ]

                if self.column == "subject":
                    self.llm_structured = self.llm.with_structured_output(Subject)

                fill_value = self._handle_no_column_data_available(
                    text_value, alternatives
                )
        return fill_value

    def _handle_no_column_data_available(
        self, text_value: str, alternatives: List[str]
    ) -> str:
        self.llm = cast(BaseChatModel, self.llm)
        try:
            for alternative_text in alternatives:
                text_value = text_value + f", {alternative_text}"
            response_structured: BaseEnum = cast(
                BaseEnum, self.llm_structured.invoke(text_value)
            )
            if self._is_subject_enum_other(response_structured):
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
                        "content": text_value,
                    },
                ]
                response = self.llm.invoke(messages)
                if response and response.content and isinstance(response.content, str):
                    return response.content.lower()
            return response_structured.enum.value.lower()
        except Exception as e:
            raise GenericException(
                "Error generating subject via LLM",
                {
                    "text_value": text_value,
                    "model": "GPT",
                    "model_name": "gpt-4o",
                    "error": str(e),
                },
            )

    def _is_subject_enum_other(self, response_structured):
        return cast(Subject, response_structured).enum == SubjectEnum.OTHER

    def _extract_words_in_parens(self, text: str) -> List[str]:
        if not text.startswith("("):
            raise ValueError("Text column does not start with a parenthesis '('")

        match = re.search(r"\((.*?)\)", text.strip())
        if not match:
            raise ValueError("Closing parenthesis ')' not found in the text")

        inside_parens = match.group(1).strip()
        words_list: List[str] = inside_parens.split()

        if words_list:
            return words_list
        else:
            raise ValueError("No words found inside the parentheses")

    def _is_wrong_type_data(
        self,
        value: AvailableTypes,
        type: Type[AvailableTypes],
    ) -> bool:
        return not isinstance(value, type)

    def _is_empty(self, row_value: str | float | None) -> bool:
        if row_value is None or self._is_nan(row_value):
            return True
        if isinstance(row_value, str):
            return row_value.strip() == ""
        return False

    def _is_nan(self, value) -> bool:
        return pd.isna(value)

    def _read_csv_with_header(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            raise GenericException("File not found", {"file_path": self.file_path})
        except pd.errors.EmptyDataError:
            raise GenericException("The file is empty", {"file_path": self.file_path})
        except pd.errors.ParserError:
            raise GenericException(
                "Error parsing the file", {"file_path": self.file_path}
            )
        except Exception as e:
            raise GenericException(
                "An unexpected error occurred",
                {"file_path": self.file_path, "error": str(e)},
            )

import os
import re
from typing import (
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)
from typing_extensions import TypedDict
import pandas as pd


from src.exceptions.generic import GenericException
from src.shared.csv.generate_csv_data import GenerateData


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


AvailableTypes = Union[int, str, bool, float]
QuestionRowTypeHints = Dict[str, Type[AvailableTypes]]


class CSVFormatter(GenerateData):
    def __init__(
        self, file_path: str, model: Literal["GPT", "GEMINI", "CLAUDE"], model_name: str
    ) -> None:
        self.file_path: str = file_path
        self.data: pd.DataFrame = self._read_csv_with_header()
        super().__init__(
            model=model,
            model_name=model_name,
        )

    def format(self):
        if self.data is None:
            raise GenericException("The file is empty.", {"file_path": self.file_path})
        for index, row in self.data.iterrows():
            text_value: bool = row.get("is_discursive", False)
            if text_value:
                continue
            filled = self._fill_missing(
                current_row=row,
                column="subject",
                current_index=index,
            )
            if not filled:
                continue
            print(f"Filled with: {filled}")

    def _fill_missing(
        self,
        column: str,
        current_row: pd.Series,
        current_index: Hashable,
    ) -> Optional[AvailableTypes]:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        type_hints: QuestionRowTypeHints = get_type_hints(QuestionRow)
        column_type = type_hints.get(column)

        if not column_type:
            raise ValueError(f"Column '{column}' not found in QuestionRow annotations.")

        fill_value: Optional[AvailableTypes] = current_row[column]
        if not self._is_empty(row_value=fill_value):
            return
        fill_value = cast(AvailableTypes, fill_value)
        if column == "subject":
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
            subject_from_text = self._extract_last_word_in_parens(text_value)
            if subject_from_text:
                fill_value = subject_from_text
        else:
            alternatives = [
                current_row.get("A", ""),
                current_row.get("B", ""),
                current_row.get("C", ""),
                current_row.get("D", ""),
                current_row.get("E", ""),
            ]
            fill_value = self._handle_no_subject_available(text_value, alternatives)
        return fill_value

    def _handle_no_subject_available(
        self,
        text_value: str,
        alternatives: List[str]
    ) -> str:
        try:
            generated_subject = self.invoke(text_value, alternatives)
            return generated_subject
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

    def _extract_last_word_in_parens(self, text: str) -> str:
        if not text.startswith("("):
            raise ValueError("Text column does not start with a parenthesis '('")

        match = re.search(r"\((.*?)\)", text.strip())
        if not match:
            raise ValueError("Closing parenthesis ')' not found in the text")

        inside_parens = match.group(1).strip()
        words_list: List[str] = inside_parens.split()

        if words_list:
            return words_list[-1]
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


base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "../../../assets")
file_path = os.path.join(assets_dir, "exported_questions.csv")
test = CSVFormatter(
    file_path=file_path,
    model="GPT",
    model_name="gpt-4o",
)
response = test.format()

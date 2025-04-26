import os

from src.shared.csv_formatter import CSV

base_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(base_dir, "../assets")
file_name = "info.csv"
file_path = os.path.join(assets_dir, file_name)
# test = CSV(
#     file_path=file_path,
# )
# test.format(
#     column="subject",
#     model="openai",
#     model_name="gpt-4o",
# )
# test.data
# breakpoint()
# file_name = "info.csv"
# file_path = os.path.join(assets_dir, file_name)
# test = CSV(
#     file_path=file_path,
# )
# test.format(
#     column="source",
#     model="openai",
#     model_name="gpt-4o",
# )
# test.data
# breakpoint()
test = CSV(
    file_path=file_path,
)
test.format(
    column="category",
    model="openai",
    model_name="gpt-4o",
)
test.data
breakpoint()

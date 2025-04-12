from typing import Optional, Required, TypedDict, Union

from .file_origin import FileOrigin, FileOriginEnum


class Payload(TypedDict):
    file_path: Required[str]
    file_origin: Optional[Union[FileOrigin, FileOriginEnum]]

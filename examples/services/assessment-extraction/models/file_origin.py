from enum import Enum
from typing import Literal

FileOrigin = Literal["local", "external"]


class FileOriginEnum(str, Enum):
    local = "local"
    external = "external"

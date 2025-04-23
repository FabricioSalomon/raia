import logging
from abc import ABC, abstractmethod
from typing import List

from src.api.schemas.questions import Payload, QuestionResponse

log = logging.getLogger(__name__)


class GenerateInterface(ABC):
    @abstractmethod
    def invoke(self, payload: Payload) -> List[QuestionResponse]:
        log.debug("Method not implemented")
        pass

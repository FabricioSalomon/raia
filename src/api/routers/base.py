import logging
from abc import ABC, abstractmethod
from typing import List

from api.schemas.questions import Payload, QuestionResponse

log = logging.getLogger(__name__)


class GenerateInterface(ABC):
    @abstractmethod
    def invoke(self, payload: Payload) -> List[QuestionResponse]:
        log.debug("Method not implemented")

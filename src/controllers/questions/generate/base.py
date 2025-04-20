from abc import ABC, abstractmethod
import logging
from typing import List
from .model import Payload, QuestionResponse

log = logging.getLogger(__name__)

class GenerateInterface(ABC):
    @abstractmethod
    def invoke(self, payload: Payload) -> List[QuestionResponse]:
        log.debug("Method not implemented")

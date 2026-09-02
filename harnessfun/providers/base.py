"""Abstract Base Class for harnessfun LLM Providers."""

from abc import ABC, abstractmethod
from typing import Callable, List
from harnessfun.models import Message, ProviderResponse


class BaseLLMProvider(ABC):
    """Provider interface for normalized model generation and model discovery."""

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: List[Callable],
        model: str,
        system_instruction: str
    ) -> ProviderResponse:
        """Translates normalized messages -> provider SDK -> normalized ProviderResponse."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Queries and returns list of available model IDs for the active project/region."""
        pass

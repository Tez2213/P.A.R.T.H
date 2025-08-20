from abc import ABC, abstractmethod
from typing import Iterable, List, Dict, Generator, Optional


Message = Dict[str, str] # {"role": "system|user|assistant", "content": str}


class BaseLLM(ABC):
    """Abstract LLM interface for chat completion."""


    def __init__(self, model: str, params: Optional[dict] = None):
        self.model = model
        self.params = params or {}


    @abstractmethod
    def chat(self, messages: List[Message]) -> str:
        """Non-streaming chat completion (returns full text)."""
        raise NotImplementedError


    @abstractmethod
    def stream(self, messages: List[Message]) -> Generator[str, None, None]:
        """Streaming chat completion (yields text chunks)."""
        raise NotImplementedError
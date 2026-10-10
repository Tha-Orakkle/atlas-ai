from collections.abc import Iterator
from typing import Protocol

from atlas_ai.llm.streaming import StreamEvent
from atlas_ai.models import ConversationMessage, LLMResponse


class LLMClient(Protocol):
    def generate(
        self,
        context: list[ConversationMessage]
    ) -> LLMResponse:
        """"Generate a response from the language model."""
        ...


class StreamingLLMClient(Protocol):
    def stream(
        self,
        context: list[ConversationMessage]
    ) -> Iterator[StreamEvent]:
        """Receive LLM response incrementally"""
        ...

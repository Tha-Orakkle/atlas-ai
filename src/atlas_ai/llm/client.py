from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from atlas_ai.llm.streaming import StreamEvent
from atlas_ai.models import ConversationMessage, LLMResponse


class LLMClient(Protocol):
    def generate(
        self,
        context: list[ConversationMessage]
    ) -> LLMResponse:
        """"Generate a complete response from the language model."""
        ...


class StreamingLLMClient(LLMClient, Protocol):
    def stream(
        self,
        context: list[ConversationMessage]
    ) -> Iterator[StreamEvent]:
        """Yield application-level events from a streamed response."""
        ...


class AsyncStreamingLLMClient(Protocol):
    def astream(
        self,
        context: list[ConversationMessage]
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronously streams events from language model."""
        ...


class AssistantLLMClient(StreamingLLMClient, Protocol):
    def astream(
        self,
        context: list[ConversationMessage]
    ) -> AsyncIterator[StreamEvent]:
        """
        Protocol for client with generate, stream and astream capabilities.
        """
        ...

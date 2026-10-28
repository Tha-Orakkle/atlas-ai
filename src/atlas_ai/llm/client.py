from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from atlas_ai.llm.streaming import StreamEvent
from atlas_ai.models import ConversationMessage, LLMResponse
from atlas_ai.observability import RequestContext


class LLMClient(Protocol):
    def generate(
        self,
        context: list[ConversationMessage],
        *,
        request_context: RequestContext | None = None
    ) -> LLMResponse:
        """"Generate a complete response from the language model."""
        ...


class StreamingLLMClient(LLMClient, Protocol):
    def stream(
        self,
        context: list[ConversationMessage],
        *,
        request_context: RequestContext | None = None
    ) -> Iterator[StreamEvent]:
        """Yield application-level events from a streamed response."""
        ...


class AsyncStreamingLLMClient(Protocol):
    def astream(
        self,
        context: list[ConversationMessage],
        *,
        request_context: RequestContext | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronously streams events from language model."""
        ...


class AssistantLLMClient(StreamingLLMClient, Protocol):
    def astream(
        self,
        context: list[ConversationMessage],
        *,
        request_context: RequestContext | None = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Protocol for client with generate, stream and astream capabilities.
        """
        ...

from collections.abc import Iterator, Sequence

from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    StreamEvent,
    TextDelta,
)
from atlas_ai.models import AssistantMessage
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor
from atlas_ai.tools.registry import TOOLS

TOOL_EXECUTOR = ToolExecutor(tools_registry=TOOLS)


class FakeStreamingLLMClient:

    def __init__(
        self,
        events: Sequence[Sequence[StreamEvent]] | None = None,
        error: Exception | None = None
    ):
        self.events = iter(events)
        self.error = error
        self.context = []

    def stream(self, context) -> Iterator[StreamEvent]:
        self.context.append(context.copy())

        if self.error is not None:
            yield StreamError(error=self.error)
            return

        yield from next(self.events)


def test_stream_response_forwards_text_deltas_and_one_response_completed():
    streams = [
        TextDelta(text="Hello "),
        TextDelta(text="world."),
        ResponseCompleted(
            message=AssistantMessage(text="Hello world.")
        )
    ]
    client = FakeStreamingLLMClient(
        events=[streams]
    )

    assistant = AssistantService(client, TOOL_EXECUTOR)

    events = list(assistant.stream_response("Say hello."))
    assert events == streams

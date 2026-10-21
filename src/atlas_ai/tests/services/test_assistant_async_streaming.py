import asyncio
from unittest.mock import Mock

from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    TextDelta,
)
from atlas_ai.models import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor


class FakeAsyncLLMClient:
    def __init__(self, events):
        self.events = iter(events)
        self.contexts = []

    async def astream(self, context):
        self.contexts.append(context.copy())

        for event in next(self.events):
            yield event


def collect(async_iterator):
    async def _collect():
        return [event async for event in async_iterator]

    return asyncio.run(_collect())


def test_astream_response_forwards_text_and_completion():
    final_message = AssistantMessage(text="Hello world.")
    client = FakeAsyncLLMClient([
        [
            TextDelta(text="Hello "),
            TextDelta(text="world."),
            ResponseCompleted(message=final_message),
        ],
    ])

    assistant = AssistantService(client, Mock(spec=ToolExecutor))

    events = collect(assistant.astream_response("Say hello."))

    assert events == [
        TextDelta(text="Hello "),
        TextDelta(text="world."),
        ResponseCompleted(message=final_message),
    ]
    assert assistant.conversation[-1] == final_message


def test_astream_response_executes_tool_and_continues():
    tool_call = ToolCall(
        call_id="call_123",
        name="calculate",
        arguments={"expression": "2 + 2"},
    )
    tool_result = ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 4},
            ),
        ],
    )
    final_message = AssistantMessage(text="The answer is 4.")

    executor = Mock(spec=ToolExecutor)
    executor.execute.return_value = tool_result

    client = FakeAsyncLLMClient([
        [
            ResponseCompleted(
                message=AssistantMessage(tool_calls=[tool_call])
            ),
        ],
        [
            TextDelta(text="The answer is 4."),
            ResponseCompleted(message=final_message),
        ],
    ])

    assistant = AssistantService(client, executor)

    events = collect(
        assistant.astream_response("What is 2 + 2?")
    )

    executor.execute.assert_called_once_with([tool_call])
    assert events == [
        TextDelta(text="The answer is 4."),
        ResponseCompleted(message=final_message),
    ]
    assert len(client.contexts) == 2
    assert client.contexts[1][-2] == AssistantMessage(
        tool_calls=[tool_call]
    )
    assert client.contexts[1][-1] == tool_result


def test_astream_response_stops_after_stream_error():
    error = StreamError(error=RuntimeError("Provider unavailable"))
    client = FakeAsyncLLMClient([[error]])
    assistant = AssistantService(client, Mock(spec=ToolExecutor))

    events = collect(assistant.astream_response("Hello."))

    assert events == [error]
    assert len(assistant.conversation) == 1

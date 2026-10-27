import logging
from collections.abc import Iterator, Sequence
from decimal import Decimal

import pytest

from atlas_ai.cost import CostCalculator, ModelPricing
from atlas_ai.errors import (
    AtlasError,
    LLMBadResponseError,
)
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    StreamEvent,
    TextDelta,
    ToolCallCompleted,
)
from atlas_ai.models import (
    AssistantMessage,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from atlas_ai.services.assistant import AssistantService


class FakeStreamingLLMClient:

    def __init__(
        self,
        events: Sequence[Sequence[StreamEvent]] | None = None,
    ):
        self.events = iter(events or [])
        self.contexts = []

    def stream(self, context) -> Iterator[StreamEvent]:
        self.contexts.append(context.copy())
        yield from next(self.events)


@pytest.fixture
def assistant_factory(tool_executor):
    def factory(events):
        client = FakeStreamingLLMClient(events)
        assistant = AssistantService(client, tool_executor)
        return assistant, client

    return factory


def test_stream_response_forwards_text_deltas_and_one_completion(
    assistant_factory
):
    events = [
        TextDelta(text="Hello "),
        TextDelta(text="world."),
        ResponseCompleted(
            message=AssistantMessage(text="Hello world.")
        )
    ]

    assistant, _ = assistant_factory([events])

    result = list(assistant.stream_response("Say hello."))
    assert result == events

    assert sum(
        isinstance(event, ResponseCompleted)
        for event in result
    ) == 1


def test_stream_response_persists_completed_conversation(
    assistant_factory
):
    final_message = AssistantMessage(text="Hello world.")
    assistant, _ = assistant_factory([
        [
            TextDelta(text="Hello world."),
            ResponseCompleted(message=final_message)
        ]
    ])

    list(assistant.stream_response("Say hello."))

    assert assistant.conversation[-2] == UserMessage(content="Say hello.")
    assert assistant.conversation[-1] == final_message


def test_stream_response_executes_tool_and_continues_streaming(
    assistant_factory,
    tool_executor
):
    tool_call = ToolCall(
        name="calculate",
        call_id="call_123",
        arguments={"expression": "2 + 2"}
    )

    tool_result = ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 4},
            )
        ]
    )

    final_message = AssistantMessage(text="The answer is 4.")

    tool_executor.execute.return_value = tool_result

    tool_call_event = [
        ToolCallCompleted(tool_call=tool_call),
        ResponseCompleted(
            message=AssistantMessage(tool_calls=[tool_call])
        )
    ]

    final_event = [
        TextDelta(text="The answer is 4."),
        ResponseCompleted(
            message=final_message
        )
    ]

    assistant, client = assistant_factory([
        tool_call_event, final_event
    ])

    events = list(assistant.stream_response("What is 2 + 2?"))

    assert events == final_event
    assert len(client.contexts) == 2

    assert client.contexts[1][-2] == AssistantMessage(
        tool_calls=[tool_call]
    )
    assert client.contexts[1][-1] == tool_result

    assert assistant.conversation[-2] == UserMessage(
        content="What is 2 + 2?"
    )
    assert assistant.conversation[-1] == final_message


def test_stream_response_forwards_stream_error_and_stops(
    assistant_factory
):
    error = StreamError(error=AtlasError("Provider unavailable"))
    assistant, _ = assistant_factory([
        [error],
    ])

    assert len(assistant.conversation) == 1

    events = list(assistant.stream_response("Hello."))

    assert events == [error]
    assert not any(
        isinstance(event, ResponseCompleted)
        for event in events
    )
    # check that the user's message is not
    # added to the conversation history
    assert len(assistant.conversation) == 1


def test_stream_response_emits_error_if_stream_has_no_completion(
    assistant_factory,
):
    partial_text_delta = TextDelta(text="Partial output")
    assistant, _ = assistant_factory([
        [partial_text_delta]
    ])

    events = list(assistant.stream_response("Hello."))

    assert len(events) == 2
    assert events[0] == partial_text_delta
    assert isinstance(events[1], StreamError)
    assert isinstance(events[1].error, LLMBadResponseError)
    assert len(assistant.conversation) == 1


def test_stream_response_removes_user_input_when_stream_raises_atlas_error(
    assistant_factory,
    tool_executor
):
    class FailingStreamingClient:
        def stream(self, context):
            raise AtlasError("Provider unavailable")
            yield

    assistant = AssistantService(
        FailingStreamingClient(),
        tool_executor,
    )

    events = list(assistant.stream_response("Hello."))

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert isinstance(events[0].error, AtlasError)
    assert len(assistant.conversation) == 1


def test_stream_response_does_not_persist_intermediate_tool_messages(
    assistant_factory,
    tool_executor,
):
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
    final_message = AssistantMessage(text="4")
    tool_executor.execute.return_value = tool_result

    assistant, _ = assistant_factory([
        [
            ResponseCompleted(
                message=AssistantMessage(tool_calls=[tool_call])
            ),
        ],
        [
            ResponseCompleted(message=final_message),
        ],
    ])

    list(assistant.stream_response("Calculate 2 + 2."))

    assert assistant.conversation == [
        assistant.conversation[0],
        UserMessage(content="Calculate 2 + 2."),
        final_message,
    ]


def test_stream_response_does_not_emit_duplicate_completion(
    assistant_factory,
):
    final_message = AssistantMessage(text="Done.")
    assistant, _ = assistant_factory([
        [
            TextDelta(text="Done."),
            ResponseCompleted(message=final_message),
        ],
    ])

    events = list(assistant.stream_response("Finish this."))

    completed_events = [
        event for event in events
        if isinstance(event, ResponseCompleted)
    ]

    assert len(completed_events) == 1
    assert completed_events[0].message == final_message


def test_stream_response_logs_request_start_and_completion(
    caplog,
    assistant_factory
):
    final_message = AssistantMessage(text="Done.")
    assistant, _ = assistant_factory([
        [
            TextDelta(text="Done."),
            ResponseCompleted(message=final_message),
        ],
    ])

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant"
    ):
        list(assistant.stream_response("Hello"))

    assert "Streaming request started | request_id=" in caplog.text
    assert "Streaming request completed | request_id=" in caplog.text


def test_stream_response_logs_failed_request(
    caplog,
    assistant_factory
):
    assistant, _ = assistant_factory([[]])

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant"
    ):
        list(assistant.stream_response("Hello."))

    assert "Streaming request started | request_id=" in caplog.text
    assert "Streaming request failed | request_id=" in caplog.text


def test_stream_response_logs_aggregate_usage_across_tool_calls(
    caplog, tool_executor
):

    events = [
        [
            ResponseCompleted(
                message=AssistantMessage(
                    tool_calls=[ToolCall(
                        call_id="call_123",
                        name="calculate",
                        arguments={"expression": "2 + 2"},
                    )]
                ),
                usage=TokenUsage(100, 20, 120),
            )
        ],
        [
            ResponseCompleted(
                message=AssistantMessage(text="4"),
                usage=TokenUsage(150, 30, 180),
            )
        ]
    ]

    client = FakeStreamingLLMClient(events)
    assistant = AssistantService(client, tool_executor)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        list(assistant.stream_response("What is 2 + 2?"))

    assert "input_tokens=250" in caplog.text
    assert "output_tokens=50" in caplog.text
    assert "total_tokens=300" in caplog.text


def test_stream_response_logs_aggregate_usage_and_cost(
    caplog, tool_executor
):
    calculator = CostCalculator(
        {
            "test-prpovider:example-model": ModelPricing(
                input_per_million=Decimal("2.00"),
                output_per_million=Decimal("8.00"),
                cached_input_per_million=Decimal("0.50"),
            )
        }
    )

    tool_call = ToolCall(
        call_id="call_123",
        name="calculate",
        arguments={"expression": "2 + 2"},
    )

    tool_executor.execute.return_value = ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 4},
            )
        ]
    )

    client = FakeStreamingLLMClient([
        [
            ResponseCompleted(
                message=AssistantMessage(tool_calls=[tool_call]),
                usage=TokenUsage(
                    1000, 100, 1100,
                    cached_input_tokens=100,
                ),
            ),
        ],
        [
            ResponseCompleted(
                message=AssistantMessage(text="4"),
                usage=TokenUsage(2000, 300, 2300),
            ),
        ],
    ])

    assistant = AssistantService(
        client,
        tool_executor,
        provider="test-prpovider",
        model="example-model",
        cost_calculator=calculator,
    )

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        events = list(assistant.stream_response("What is 2 + 2?"))

    completion = next(
        event for event in events
        if isinstance(event, ResponseCompleted)
    )

    assert completion.usage == TokenUsage(
        input_tokens=3000,
        output_tokens=400,
        total_tokens=3400,
        cached_input_tokens=100,
        reasoning_tokens=0,
    )
    assert "Interaction usage" in caplog.text
    assert "cached_input_tokens=100" in caplog.text
    assert "estimated_cost_usd=0.00905" in caplog.text

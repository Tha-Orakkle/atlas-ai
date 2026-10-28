import asyncio
import logging
from decimal import Decimal
from unittest.mock import Mock

from atlas_ai.cost import CostCalculator, ModelPricing
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    TextDelta,
)
from atlas_ai.models import (
    AssistantMessage,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)
from atlas_ai.observability import RequestContext
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor


def _log(task):
    return (
        f"{task} (async-streaming) | interaction_id="
    )


class FakeAsyncLLMClient:
    def __init__(self, events):
        self.events = iter(events)
        self.contexts = []
        self.request_contexts = []

    async def astream(self, context, *, request_context):
        self.contexts.append(context.copy())
        self.request_contexts.append(request_context)

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
    args, kwargs = executor.execute.call_args

    assert args[0] == [tool_call]
    assert isinstance(kwargs["request_context"], RequestContext)

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


def test_astream_response_logs_aggregate_usage_across_tool_calls(
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

    client = FakeAsyncLLMClient(events)
    assistant = AssistantService(client, tool_executor)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        collect(assistant.astream_response("What is 2 + 2?"))

    assert "input_tokens=250" in caplog.text
    assert "output_tokens=50" in caplog.text
    assert "total_tokens=300" in caplog.text


def test_astream_response_logs_aggregate_usage_and_cost(
    caplog, tool_executor
):
    calculator = CostCalculator(
        {
            "test-provider:example-model": ModelPricing(
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

    client = FakeAsyncLLMClient([
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
        provider="test-provider",
        model="example-model",
        cost_calculator=calculator,
    )

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        events = collect(
            assistant.astream_response("What is 2 + 2?")
        )

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


def test_astream_tool_loop_preserves_interaction_id_and_changes_request_id(
    tool_executor,
):
    tool_call = ToolCall(
        call_id="call_123",
        name="calculate",
        arguments={"expression": "2 + 2"},
    )
    tool_executor.execute.return_value = ToolResultMessage(
        results=[
            ToolResult(call_id="call_123", result={"result": 4})
        ]
    )

    client = FakeAsyncLLMClient([
        [
            ResponseCompleted(
                message=AssistantMessage(tool_calls=[tool_call])
            )
        ],
        [
            ResponseCompleted(message=AssistantMessage(text="4"))
        ],
    ])
    assistant = AssistantService(client, tool_executor)

    collect(assistant.astream_response("Calculate 2 + 2."))

    first, second = client.request_contexts
    assert first.interaction_id == second.interaction_id
    assert first.request_id != second.request_id

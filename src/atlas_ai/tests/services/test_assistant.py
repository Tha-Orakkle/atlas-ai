import logging
from decimal import Decimal

import pytest

from atlas_ai.cost import CostCalculator, ModelPricing
from atlas_ai.errors import AtlasError
from atlas_ai.models import (
    AssistantMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
)
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor
from atlas_ai.tools.registry import TOOLS


def _log(task):
    return (
        f"{task} | interaction_id="
    )


class FakeLLMClient:
    def __init__(self, responses=None, error=None):
        self.responses = iter(responses or [])
        self.error = error
        self.contexts = []
        self.request_contexts = []

    def generate(self, context, *, request_context):
        self.contexts.append(context.copy())
        self.request_contexts.append(request_context)

        if self.error is not None:
            raise self.error

        return next(self.responses)


def test_generate_response_logs_request_start_and_completion(
    caplog, tool_executor
):
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(text="Hello")
            )
        ]
    )
    assistant = AssistantService(client, tool_executor)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant"
    ):
        result = assistant.generate_response("Hello")

    assert result == "Hello"
    assert _log("Interaction started") in caplog.text
    assert _log("Preparing LLM request") in caplog.text
    assert _log("Interaction completed") in caplog.text


def test_generate_response_persists_only_completed_assistant_turn(
    tool_executor
):
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            call_id="call_123",
                            name="calculate",
                            arguments={"expression": "2 + 2"},
                        )
                    ]
                )
            ),
            LLMResponse(
                message=AssistantMessage(text="4")
            ),
        ]
    )
    assistant = AssistantService(client, tool_executor)

    result = assistant.generate_response("What is 2 + 2?")

    assert result == "4"
    assert isinstance(assistant.conversation[-1], AssistantMessage)
    assert assistant.conversation[-1].text == "4"
    assert assistant.conversation[-1].tool_calls is None
    assert not any(
        isinstance(message, ToolResultMessage)
        for message in assistant.conversation
    )


def test_generate_response_passes_tool_call_and_result_to_next_llm_call():
    tool_executor = ToolExecutor(tools_registry=TOOLS)

    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            call_id="call_123",
                            name="calculate",
                            arguments={"expression": "2 + 2"},
                        )
                    ]
                )
            ),
            LLMResponse(
                message=AssistantMessage(text="The answer is 4.")
            ),
        ]
    )
    assistant = AssistantService(client, tool_executor)

    result = assistant.generate_response("What is 2 + 2?")

    assert result == "The answer is 4."
    assert len(client.contexts) == 2

    second_context = client.contexts[1]

    assert isinstance(second_context[-2], AssistantMessage)
    assert second_context[-2].tool_calls == [
        ToolCall(
            call_id="call_123",
            name="calculate",
            arguments={"expression": "2 + 2"},
        )
    ]

    assert second_context[-1] == ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 4},
            )
        ]
    )


def test_generate_response_removes_user_input_when_llm_fails(
    caplog, tool_executor
):
    error = AtlasError("LLM unavailable")
    assistant = AssistantService(
        FakeLLMClient(error=error),
        tool_executor,
    )

    with pytest.raises(AtlasError), caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        assistant.generate_response("Hello")

    assert len(assistant.conversation) == 1
    assert _log("Interaction started") in caplog.text
    assert _log("Preparing LLM request") in caplog.text
    assert _log("Interaction failed") in caplog.text


def test_generate_response_logs_aggregate_usage_across_tool_calls(
    caplog, tool_executor
):
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            call_id="call_123",
                            name="calculate",
                            arguments={"expression": "2 + 2"},
                        )
                    ]
                ),
                usage=TokenUsage(100, 20, 120),
            ),
            LLMResponse(
                message=AssistantMessage(text="4"),
                usage=TokenUsage(150, 30, 180),
            ),
        ]
    )
    assistant = AssistantService(client, tool_executor)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        assistant.generate_response("What is 2 + 2?")

    assert "input_tokens=250" in caplog.text
    assert "output_tokens=50" in caplog.text
    assert "total_tokens=300" in caplog.text


def test_generate_response_logs_estimated_interaction_cost(
    caplog,
    tool_executor,
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

    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(text="Hello"),
                usage=TokenUsage(1000, 300, 1300),
            )
        ]
    )

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
        assistant.generate_response("Hello")

    assert "Interaction cost estimate" in caplog.text
    assert "model=test-provider:example-model" in caplog.text
    assert "estimated_cost_usd=0.0044" in caplog.text


def test_cost_is_unavailable_without_pricing_configuration(
    caplog,
    tool_executor,
):
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(text="Hello"),
                usage=TokenUsage(1000, 300, 1300),
            )
        ]
    )

    assistant = AssistantService(
        client,
        tool_executor,
        model="example-model",
        cost_calculator=CostCalculator({}),
    )

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        assistant.generate_response("Hello")

    assert "Interaction cost unavailable" in caplog.text
    assert "estimated_cost_usd=" not in caplog.text


def test_generate_response_logs_all_aggregate_usage_fields(
    caplog, tool_executor
):
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=30,
        reasoning_tokens=5,
    )
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(text="Hello"),
                usage=usage,
            )
        ]
    )
    assistant = AssistantService(client, tool_executor)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        assert assistant.generate_response("Hello") == "Hello"

    assert "Interaction usage" in caplog.text
    assert "input_tokens=100" in caplog.text
    assert "output_tokens=20" in caplog.text
    assert "total_tokens=120" in caplog.text
    assert "cached_input_tokens=30" in caplog.text
    assert "reasoning_tokens=5" in caplog.text


def test_generate_response_logs_unavailable_usage(
    caplog, tool_executor
):
    client = FakeLLMClient(
        responses=[
            LLMResponse(message=AssistantMessage(text="Hello"))
        ]
    )
    assistant = AssistantService(client, tool_executor)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.services.assistant",
    ):
        assert assistant.generate_response("Hello") == "Hello"

    assert "Interaction usage unavailable" in caplog.text


def test_generate_response_logs_cost_for_entire_tool_interaction(
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

    tool_executor.execute.return_value = ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 4},
            )
        ]
    )

    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            call_id="call_123",
                            name="calculate",
                            arguments={"expression": "2 + 2"},
                        )
                    ]
                ),
                usage=TokenUsage(
                    1000, 100, 1100,
                    cached_input_tokens=100,
                ),
            ),
            LLMResponse(
                message=AssistantMessage(text="4"),
                usage=TokenUsage(2000, 300, 2300),
            ),
        ]
    )

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
        assert assistant.generate_response("What is 2 + 2?") == "4"

    assert "input_tokens=3000" in caplog.text
    assert "output_tokens=400" in caplog.text
    assert "total_tokens=3400" in caplog.text
    assert "cached_input_tokens=100" in caplog.text
    assert "estimated_cost_usd=0.00905" in caplog.text


def test_cost_estimation_failure_does_not_fail_interaction(
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

    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(text="Hello"),
                usage=TokenUsage(
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                    cached_input_tokens=200,
                ),
            )
        ]
    )

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
        result = assistant.generate_response("Hello")

    assert result == "Hello"
    assert "Interaction cost estimation failed" in caplog.text
    assert _log("Interaction completed") in caplog.text


def test_tool_loop_reuses_interaction_id_but_creates_new_request_ids(
    tool_executor,
):
    client = FakeLLMClient(
        responses=[
            LLMResponse(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            call_id="call_123",
                            name="calculate",
                            arguments={"expression": "2 + 2"},
                        )
                    ]
                )
            ),
            LLMResponse(message=AssistantMessage(text="4")),
        ]
    )
    assistant = AssistantService(client, tool_executor)

    assert assistant.generate_response("What is 2 + 2?") == "4"

    first, second = client.request_contexts

    assert first.interaction_id == second.interaction_id
    assert first.request_id != second.request_id

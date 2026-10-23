import json
from types import SimpleNamespace

from atlas_ai.models import (
    AssistantMessage,
    DeveloperMessage,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)


def test_translates_application_context_to_openai_input(openai_client):
    context = [
        DeveloperMessage(content="You are Atlas."),
        UserMessage(content="What is 2 + 2?"),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    call_id="call_123",
                    name="calculate",
                    arguments={"expression": "2 + 2"},
                )
            ]
        ),
        ToolResultMessage(
            results=[
                ToolResult(
                    call_id="call_123",
                    result={"result": 4},
                )
            ]
        ),
    ]

    result = openai_client._to_openai_input(context)

    assert result == [
        {
            "role": "developer",
            "content": "You are Atlas.",
        },
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "calculate",
            "arguments": json.dumps({"expression": "2 + 2"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps({"result": 4}),
        },
    ]


def test_translates_final_assistant_message(openai_client):
    result = openai_client._to_openai_input(
        [AssistantMessage(text="The answer is 4.")]
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
        }
    ]


def test_translates_openai_tool_call_response_to_application_response(
    openai_client
):
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_123",
                name="calculate",
                arguments='{"expression": "2 + 2"}',
            )
        ],
        output_text="",
    )

    result = openai_client._from_openai_response(response)

    assert result.message == AssistantMessage(
        tool_calls=[
            ToolCall(
                call_id="call_123",
                name="calculate",
                arguments={"expression": "2 + 2"},
            )
        ]
    )


def test_translates_openai_final_response_to_application_response(
    openai_client
):
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[],
            )
        ],
        output_text="The answer is 4.",
    )

    result = openai_client._from_openai_response(response)

    assert result.message == AssistantMessage(
        text="The answer is 4."
    )


def test_generate_reconstructs_tool_call_and_result_for_next_request(
    openai_client
):
    openai_client.client.responses.create.return_value = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[],
            )
        ],
        output_text="The answer is 4.",
    )

    context = [
        UserMessage(content="What is 2 + 2?"),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    call_id="call_123",
                    name="calculate",
                    arguments={"expression": "2 + 2"},
                )
            ]
        ),
        ToolResultMessage(
            results=[
                ToolResult(
                    call_id="call_123",
                    result={"result": 4},
                )
            ]
        ),
    ]

    result = openai_client.generate_response(context)

    openai_client.client.responses.create.assert_called_once_with(
        model="gpt-test",
        input=[
            {
                "role": "user",
                "content": "What is 2 + 2?",
            },
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "calculate",
                "arguments": json.dumps({"expression": "2 + 2"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": json.dumps({"result": 4}),
            },
        ],
        tools=openai_client.tools_schema,
    )

    assert result.message == AssistantMessage(
        text="The answer is 4."
    )


def test_translates_openai_usage_to_token_usage(openai_client):
    response = SimpleNamespace(
        output=[],
        output_text="Hello.",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            input_tokens_details=SimpleNamespace(
                cached_tokens=30,
            ),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=5,
            ),
        ),
    )

    result = openai_client._from_openai_response(response)

    assert result.usage == TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=30,
        reasoning_tokens=5,
    )


def test_usage_is_none_when_provider_does_not_return_it(openai_client):
    response = SimpleNamespace(
        output=[],
        output_text="Hello.",
    )

    result = openai_client._from_openai_response(response)

    assert result.usage is None


def test_usage_details_default_to_zero_when_absent(openai_client):
    response = SimpleNamespace(
        output=[],
        output_text="Hello.",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        ),
    )

    result = openai_client._from_openai_response(response)

    assert result.usage == TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=0,
        reasoning_tokens=0,
    )

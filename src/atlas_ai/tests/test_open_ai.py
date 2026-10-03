import json
from types import SimpleNamespace
from unittest.mock import Mock

from atlas_ai.llm.open_ai import OpenAIClient
from atlas_ai.models import (
    AssistantMessage,
    DeveloperMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)


def make_client():
    client = object.__new__(OpenAIClient)
    client.model = "gpt-test"
    client.tools_schema = [{"type": "function", "name": "calculate"}]
    client.client = Mock()
    return client


def test_translates_application_context_to_openai_input():
    client = make_client()

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

    result = client._to_openai_input(context)

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


def test_translates_final_assistant_message():
    client = make_client()

    result = client._to_openai_input(
        [AssistantMessage(text="The answer is 4.")]
    )

    assert result == [
        {
            "role": "assistant",
            "content": "The answer is 4.",
        }
    ]


def test_translates_openai_tool_call_response_to_application_response():
    client = make_client()

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

    result = client._from_openai_response(response)

    assert result.message == AssistantMessage(
        tool_calls=[
            ToolCall(
                call_id="call_123",
                name="calculate",
                arguments={"expression": "2 + 2"},
            )
        ]
    )


def test_translates_openai_final_response_to_application_response():
    client = make_client()

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[],
            )
        ],
        output_text="The answer is 4.",
    )

    result = client._from_openai_response(response)

    assert result.message == AssistantMessage(
        text="The answer is 4."
    )


def test_generate_reconstructs_tool_call_and_result_for_next_request():
    client = make_client()
    client.client.responses.create.return_value = SimpleNamespace(
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

    result = client.generate_response(context)

    client.client.responses.create.assert_called_once_with(
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
        tools=client.tools_schema,
    )

    assert result.message == AssistantMessage(
        text="The answer is 4."
    )

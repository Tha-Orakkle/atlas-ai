import logging

import pytest
from unittest.mock import Mock

from atlas_ai.models import ToolCall, ToolResult, ToolResultMessage
from atlas_ai.tools.executor import ToolExecutor


@pytest.fixture
def tool_function():
    return Mock(return_value={"result": 42})


@pytest.fixture
def tools_registry(tool_function):
    return {
        "calculate": {
            "function": tool_function,
            "schema": {
                "type": "function",
                "name": "calculate",
                "description": "Calculate an expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"],
                    "additionalProperties": False,
                },
            },
        }
    }


@pytest.fixture
def executor(tools_registry):
    return ToolExecutor(tools_registry=tools_registry)


def make_tool_call(
    name="calculate",
    arguments=None,
    call_id="call_123",
):
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments or {"expression": "2 + 2"},
    )


def test_executes_valid_tool(executor, tool_function):
    tool_call = make_tool_call()

    result = executor.execute([tool_call])

    tool_function.assert_called_once_with(expression="2 + 2")
    assert result == ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 42},
            )
        ]
    )


def test_passes_multiple_tool_calls_and_preserves_call_ids(
    executor,
    tool_function,
):
    second_tool = Mock(return_value={"result": 10})
    executor.tools_registry["calculate_2"] = {
        "function": second_tool,
        "schema": {},
    }

    tool_calls = [
        make_tool_call(call_id="call_123"),
        make_tool_call(
            name="calculate_2",
            arguments={"expression": "10"},
            call_id="call_456",
        ),
    ]

    result = executor.execute(tool_calls)

    assert result == ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"result": 42},
            ),
            ToolResult(
                call_id="call_456",
                result={"result": 10},
            ),
        ]
    )


def test_unknown_tool_returns_application_level_error_result(executor):
    result = executor.execute(
        [make_tool_call(name="unknown_tool")]
    )

    assert result == ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"error": "Unknown tool unknown_tool"},
            )
        ]
    )


def test_unknown_tool_is_not_executed(executor, tool_function):
    executor.execute([make_tool_call(name="unknown_tool")])
    tool_function.assert_not_called()


def test_unknown_tool_is_logged(executor, caplog):
    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.tools.executor",
    ):
        executor.execute([make_tool_call(name="unknown_tool")])

    assert "Executing tool | tool=unknown_tool" in caplog.text
    assert "Tool not found | tool=unknown_tool" in caplog.text


def test_tool_execution_failure_returns_application_level_error(
    tools_registry,
):
    failing_tool = Mock(
        side_effect=RuntimeError("database password leaked")
    )
    tools_registry["calculate"]["function"] = failing_tool

    executor = ToolExecutor(tools_registry=tools_registry)

    result = executor.execute([make_tool_call()])

    failing_tool.assert_called_once_with(expression="2 + 2")
    assert result == ToolResultMessage(
        results=[
            ToolResult(
                call_id="call_123",
                result={"error": "Tool failed: database password leaked."},
            )
        ]
    )


def test_tool_execution_failure_is_logged(
    tools_registry,
    caplog,
):
    failing_tool = Mock(side_effect=RuntimeError("secret tool failed"))
    tools_registry["calculate"]["function"] = failing_tool
    executor = ToolExecutor(tools_registry=tools_registry)

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.tools.executor",
    ):
        executor.execute([make_tool_call()])

    assert "Executing tool | tool=calculate" in caplog.text
    assert "Tool failed | tool=calculate" in caplog.text
    assert "Tool completed | tool=calculate" not in caplog.text


def test_raw_tool_arguments_are_not_logged(executor, caplog):
    secret = "super-secret-api-key"
    tool_call = make_tool_call(
        arguments={"expression": secret}
    )

    with caplog.at_level(
        logging.INFO,
        logger="atlas_ai.tools.executor",
    ):
        executor.execute([tool_call])

    assert "Executing tool | tool=calculate" in caplog.text
    assert "Tool completed | tool=calculate" in caplog.text
    assert secret not in caplog.text
    assert "api_key" not in caplog.text

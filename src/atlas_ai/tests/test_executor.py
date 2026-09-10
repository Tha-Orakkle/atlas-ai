import json
import logging
import pytest

from unittest.mock import Mock
from types import SimpleNamespace
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
                        "expression": {
                            "type": "string"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False
                }
            }
        }
    }


@pytest.fixture
def executor(tools_registry):
    return ToolExecutor(
        tools_registry=tools_registry
    )


def make_tool_call(
    name="calculate",
    arguments='{"expression": "2 + 2"}',
    call_id="call_123"
):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments,
        call_id=call_id
    )


def test_executes_valid_tool(executor, tool_function):
    tool_call = make_tool_call()

    result = executor.execute([tool_call])

    tool_function.assert_called_once_with(
        expression="2 + 2"
    )

    assert len(result) == 1


def test_parses_and_passes_tool_arguments(
    executor,
    tool_function,
):
    tool_call = make_tool_call(
        arguments='{"expression": "10 * 5"}'
    )

    executor.execute([tool_call])

    tool_function.assert_called_once_with(
        expression="10 * 5"
    )


def test_converts_tool_result_to_function_call_output(
    executor
):
    tool_call = make_tool_call()
    result = executor.execute([tool_call])

    assert result == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps({"result": 42})
        }
    ]


def test_unknown_tool_returns_error_output(executor):
    tool_call = make_tool_call(
        name="unknown_tool"
    )

    result = executor.execute([tool_call])

    assert result == [
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps({
                "error": "Unknown tool: unknown_tool"
            })
        }
    ]


def test_unknown_tool_is_not_executed(executor, tool_function):
    tool_call = make_tool_call(
        name="unknown_tool"
    )

    executor.execute([tool_call])

    tool_function.assert_not_called()


def test_unknown_tool_is_logged(executor, caplog):
    tool_call = make_tool_call(
        name="unknown_tool"
    )
    with caplog.at_level(logging.INFO, logger="atlas_ai.tools.executor"):
        executor.execute([tool_call])

    assert "Executing tool | tool=unknown_tool" in caplog.text
    assert "Tool not found | tool=unknown_tool" in caplog.text


def test_tool_execution_failure_returns_safe_error(tools_registry):

    failing_tool = Mock(
        side_effect=RuntimeError("database password leaked")
    )
    tools_registry["calculate"]["function"] = failing_tool

    executor = ToolExecutor(
        tools_registry=tools_registry
    )

    tool_call = make_tool_call()

    result = executor.execute([tool_call])

    failing_tool.assert_called_once()

    assert result == [
        {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps({
                "error": "Tool execution failed."
            })
        }
    ]


def test_tool_execution_failure_is_logged(
    tools_registry,
    caplog
):
    failing_tool = Mock(
        side_effect=RuntimeError("secret tool failed")
    )
    tools_registry["calculate"]["function"] = failing_tool

    executor = ToolExecutor(
        tools_registry=tools_registry
    )

    tool_call = make_tool_call()

    with caplog.at_level(logging.INFO, logger="atlas_ai.tools.executor"):
        executor.execute([tool_call])

    assert "Executing tool | tool=calculate" in caplog.text
    assert "Tool execution failed | tool=calculate" in caplog.text
    assert "Tool completed | tool=calculate" in caplog.text


def test_raw_tool_arguments_are_not_logged(executor, caplog):
    secret = "super-secret-api-key"

    tool_call = make_tool_call(
        arguments=json.dumps({"expression": secret})
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.tools.executor"):
        executor.execute([tool_call])

    assert "Executing tool | tool=calculate" in caplog.text
    assert "Tool completed | tool=calculate" in caplog.text
    assert secret not in caplog.text
    assert "api_key" not in caplog.text


def test_non_tool_items_are_ignored(executor, tool_function):
    response_output = [
        SimpleNamespace(
            type="message",
            content="Hello world!"
        ),
        make_tool_call()
    ]

    result = executor.execute(response_output)

    tool_function.assert_called_once_with(
        expression="2 + 2"
    )
    assert len(result) == 1

import json
import logging
import pytest

from types import SimpleNamespace


from atlas_ai.errors import AtlasError
from atlas_ai.services.assistant import AssistantService


class FakeLLMClient:
    def __init__(self, responses=None, error=None):
        self.responses = iter(responses or [])
        self.error = error

    def generate(self, context):
        if self.error is not None:
            raise self.error
        return next(self.responses)


def make_response(output_text="success", output=None):
    return SimpleNamespace(
        output_text=output_text,
        output=output or [],
    )


def test_generate_response_logs_request_start_and_completion(caplog):
    client = FakeLLMClient(
        responses=[make_response(output_text="Hello")]
    )
    assistant = AssistantService(client)

    with caplog.at_level(logging.INFO, logger="atlas_ai.services.assistant"):
        result = assistant.generate_response("Hello")

    assert result == "Hello"
    assert "Request started | request_id=" in caplog.text
    assert "Calling LLM | request_id=" in caplog.text
    assert "Request completed | request_id=" in caplog.text


def test_generate_response_logs_failed_request_exception(caplog):
    error = AtlasError("LLM unavailable")
    assistant = AssistantService(FakeLLMClient(error=error))

    with pytest.raises(AtlasError):
        with caplog.at_level(logging.INFO, logger="atlas_ai.services.assistant"):
            assistant.generate_response("Hello")

    assert "Request started | request_id=" in caplog.text
    assert "Calling LLM | request_id=" in caplog.text
    assert "Request failed | request_id=" in caplog.text


def test_tool_execution_is_logged_without_arguments(caplog):
    secret = "super-secret-api-key"
    tool = {
        "schema": {"type": "function", "name": "test_tool"},
        "function": lambda **kwargs: {"result": "ok"},
    }

    assistant = AssistantService(FakeLLMClient())
    assistant.tools_registry = {"test_tool": tool}
    response_item = SimpleNamespace(
        type="function_call",
        name="test_tool",
        call_id="call-1",
        arguments=json.dumps({"api_key": secret}),
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.services.assistant"):
        assistant.execute_tools([response_item])

    assert "Executing tool | tool=test_tool" in caplog.text
    assert "Tool completed | tool=test_tool" in caplog.text
    assert secret not in caplog.text
    assert "api_key" not in caplog.text


def test_tool_execution_logs_unknown_tool(caplog):
    assistant = AssistantService(FakeLLMClient())
    assistant.tools_registry = {}
    response_item = SimpleNamespace(
        type="function_call",
        name="test_tool",
        call_id="call-1",
        arguments=json.dumps({})
    )

    with caplog.at_level(logging.INFO, logger="atlas_ai.services.assistant"):
        assistant.execute_tools([response_item])

    assert "Executing tool | tool=test_tool" in caplog.text
    assert "Tool execution failed. Tool not found. | tool=test_tool" in caplog.text

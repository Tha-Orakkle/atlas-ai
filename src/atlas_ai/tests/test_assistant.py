import json
import logging
import pytest

from types import SimpleNamespace

from atlas_ai.errors import AtlasError
from atlas_ai.services.assistant import AssistantService
from atlas_ai.tools.executor import ToolExecutor
from atlas_ai.tools.registry import TOOLS

TOOL_EXECUTOR = ToolExecutor(tools_registry=TOOLS)


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
    assistant = AssistantService(client, TOOL_EXECUTOR)

    with caplog.at_level(logging.INFO, logger="atlas_ai.services.assistant"):
        result = assistant.generate_response("Hello")

    assert result == "Hello"
    assert "Request started | request_id=" in caplog.text
    assert "Calling LLM | request_id=" in caplog.text
    assert "Request completed | request_id=" in caplog.text


def test_generate_response_logs_failed_request_exception(caplog):
    error = AtlasError("LLM unavailable")
    assistant = AssistantService(
        FakeLLMClient(error=error),
        TOOL_EXECUTOR
    )

    with pytest.raises(AtlasError):
        with caplog.at_level(
            logging.INFO,
            logger="atlas_ai.services.assistant"
        ):
            assistant.generate_response("Hello")

    assert "Request started | request_id=" in caplog.text
    assert "Calling LLM | request_id=" in caplog.text
    assert "Request failed | request_id=" in caplog.text

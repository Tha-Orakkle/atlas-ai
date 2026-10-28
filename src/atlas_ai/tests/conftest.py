from unittest.mock import MagicMock

import pytest

from atlas_ai.llm.open_ai import OpenAIClient
from atlas_ai.tools.executor import ToolExecutor


@pytest.fixture
def openai_client():
    client = object.__new__(OpenAIClient)
    client.provider = "openai"
    client.model = "gpt-test"
    client.tools_schema = [{"type": "function", "name": "calculate"}]
    client.client = MagicMock()
    client.async_client = MagicMock()
    return client


@pytest.fixture
def tool_executor():
    return MagicMock(spec=ToolExecutor)

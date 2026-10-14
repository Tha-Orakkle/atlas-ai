from unittest.mock import Mock

import pytest

from atlas_ai.llm.open_ai import OpenAIClient


@pytest.fixture
def openai_client():
    client = object.__new__(OpenAIClient)
    client.model = "gpt-test"
    client.tools_schema = [{"type": "function", "name": "calculate"}]
    client.client = Mock()
    return client

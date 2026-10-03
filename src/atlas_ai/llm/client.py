from typing import Protocol

from atlas_ai.models import ConversationMessage, LLMResponse


class LLMClient(Protocol):
    def generate(
        self,
        context: list[ConversationMessage]
    ) -> LLMResponse:
        """"Generate a response from the language model."""
        ...

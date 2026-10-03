from typing import Protocol


class LLMClient(Protocol):
    def generate(self, context: list[dict]):
        """"Generate a response from the language model."""
        ...

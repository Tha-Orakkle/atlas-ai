from typing import Protocol
from openai import (
    AuthenticationError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
    OpenAI
)
from atlas_ai.errors import (
    LLMBadRequestError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMConnectionError,
    LLMAuthenticationError,
)
from atlas_ai.tools.registry import TOOLS
from atlas_ai.reliability.retry import retry


class LLMClient(Protocol):
    def generate(self, context: list[dict]):
        """"Generate a response from the language model."""
        ...


class OpenAIClient:
    """
    Client communicate with the OpenAI API using the
    OpenAI SDK.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initializes an openai client
        """
        self.client = OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.tools_schema = [tool["schema"] for tool in TOOLS.values()]

    def generate_response(self, context: list[dict]):
        """
        Communicates with the OpenAI responses API.
        Args:
            - context (list): conversation context.
        """
        try:
            return self.client.responses.create(
                model=self.model,
                input=context,
                tools=self.tools_schema
            )

        except RateLimitError as exc:
            retry_after = exc.response.headers.get("Retry-After")
            raise LLMRateLimitError(
                "The LLM provider rate limit was exceeded.",
                retry_after=retry_after
            ) from exc

        except APITimeoutError as exc:
            raise LLMTimeoutError(
                "Request to LLM timed out."
            ) from exc

        except APIConnectionError as exc:
            raise LLMConnectionError(
                "Unable to connect to the LLM provider."
            ) from exc

        except AuthenticationError as exc:
            raise LLMAuthenticationError(
                "Unable to authenticate with the LLM provider."
            ) from exc

        except BadRequestError as exc:
            raise LLMBadRequestError(
                f"Invalid request to LLM provider: {exc}"
            ) from exc

        except APIStatusError as exc:
            raise LLMBadRequestError(
                f"LLM provider returned an error: {exc}"
            ) from exc

    def generate(self, context: list[dict]):
        """
        Surround communication with the OpenAI responses API
        with the Atlas retry policy.
        Args:
            - context (list): conversation history,
              function calls and function call outputs.
        """
        return retry(
            lambda: self.generate_response(context)
        )

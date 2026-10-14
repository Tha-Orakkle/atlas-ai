import json
import logging
from collections.abc import Iterator
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from openai.types.responses.response import Response

from atlas_ai.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBadResponseError,
    LLMConnectionError,
    LLMIncompleteResponseError,
    LLMRateLimitError,
    LLMResponseConsistencyError,
    LLMTimeoutError,
)
from atlas_ai.llm.client import StreamingLLMClient
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    StreamEvent,
    TextDelta,
    ToolCallCompleted,
)
from atlas_ai.models import (
    AssistantMessage,
    ConversationMessage,
    DeveloperMessage,
    LLMResponse,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from atlas_ai.reliability.retry import retry

logger = logging.getLogger(__name__)


class OpenAIClient(StreamingLLMClient):
    """
    Client communicate with the OpenAI API using the
    OpenAI SDK.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        tool_schema: list[dict[str, Any]]
    ):
        """
        Initializes an openai client
        """
        self.client = OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.tools_schema = tool_schema

    def _build_message_input(
        self,
        role: str,
        content: str
    ) -> dict[str, str]:

        """
        Build input for developer, user and
        model non-tool call messages.
        """

        return {
            "role": role,
            "content": content
        }

    def _build_openai_tool_call(
        self,
        tool_calls: list[ToolCall]
    ) -> list[dict]:

        """
        Build OpenAI tool call requests.
        """

        return [
            {
                "type": "function_call",
                "call_id": tool.call_id,
                "name": tool.name,
                "arguments": json.dumps(tool.arguments)
            }
            for tool in tool_calls
        ]

    def _build_tool_result(
        self,
        results: list[ToolResult]
    ) -> list[dict]:

        return [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": json.dumps(result.result)
            }
            for result in results
        ]

    def _to_openai_input(
        self,
        context: list[ConversationMessage]
    ) -> list[dict[str, Any]]:
        """
        Translate conversation history to model-ready input.
        """

        openai_input = []

        for message in context:
            if isinstance(message, DeveloperMessage):
                openai_input.append(
                    self._build_message_input(
                        role="developer", content=message.content
                    )
                )

            elif isinstance(message, UserMessage):
                openai_input.append(
                    self._build_message_input(
                        role="user", content=message.content
                    )
                )

            elif isinstance(message, AssistantMessage):
                if message.tool_calls:
                    openai_input.extend(
                        self._build_openai_tool_call(
                            message.tool_calls
                        )
                    )

                else:
                    openai_input.append(
                        self._build_message_input(
                            role="assistant", content=message.text
                        )
                    )

            elif isinstance(message, ToolResultMessage):
                openai_input.extend(
                    self._build_tool_result(
                        message.results
                    )
                )

        return openai_input

    def _from_openai_response(
        self,
        response: Response
    ) -> LLMResponse:
        """
        Translate model response to application-level response.
        """

        tool_calls = [
            ToolCall(
                call_id=item.call_id,
                name=item.name,
                arguments=json.loads(item.arguments)
            )
            for item in response.output if item.type == "function_call"
        ]

        if tool_calls:
            assistant_message = AssistantMessage(
                tool_calls=tool_calls
            )
        else:
            assistant_message = AssistantMessage(
                text=response.output_text
            )

        return LLMResponse(
            message=assistant_message
        )

    def generate_response(
        self,
        context: list[ConversationMessage]
    ) -> LLMResponse:
        """
        Communicates with the OpenAI responses API.
        Args:
            - context (list): conversation context.
        """
        try:
            openai_input = self._to_openai_input(context)

            logger.info("Calling OpenAI model")

            response = self.client.responses.create(
                model=self.model,
                input=openai_input,
                tools=self.tools_schema
            )

            return self._from_openai_response(response)

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

    def generate(
        self,
        context: list[ConversationMessage]
    ) -> LLMResponse:
        """
        Communicate with OpenAI model.
        Args:
            - context: Contains conversation history,
            tool call requests and results.
        """
        return retry(
            lambda: self.generate_response(context)
        )

    def stream(
        self,
        context: list[ConversationMessage]
    ) -> Iterator[StreamEvent]:

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        try:
            openai_input = self._to_openai_input(context)

            logger.info("Starting OpenAI streaming request")

            with self.client.responses.stream(
                model=self.model,
                input=openai_input,
                tools=self.tools_schema
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        text_parts.append(event.delta)
                        yield TextDelta(text=event.delta)

                    elif (
                        event.type == "response.output_item.done"
                        and event.item.type == "function_call"
                    ):
                        item = event.item
                        tool_call = ToolCall(
                            name=item.name,
                            call_id=item.call_id,
                            arguments=json.loads(item.arguments)
                        )
                        tool_calls.append(tool_call)
                        yield ToolCallCompleted(tool_call=tool_call)

                response = stream.get_final_response()

                if response.status != "completed":
                    yield StreamError(
                        error=LLMIncompleteResponseError(
                            "The LLM stream did not complete successfully."
                        )
                    )
                    return

                if tool_calls:
                    final_message = AssistantMessage(
                        tool_calls=tool_calls
                    )
                else:
                    final_message = self._from_openai_response(
                        response=response
                    ).message

                    streamed_text = "".join(text_parts)

                    if final_message.text != streamed_text:
                        yield StreamError(
                            error=LLMResponseConsistencyError(
                                "Streamed text did not match "
                                "the final response."
                            )
                        )
                        return

                yield ResponseCompleted(message=final_message)

        except RateLimitError as exc:
            retry_after = exc.response.headers.get("Retry-After")
            yield StreamError(
                error=LLMRateLimitError(
                    "The LLM provider rate limit was exceeded.",
                    retry_after=retry_after
                )
            )

        except APITimeoutError:
            yield StreamError(
                error=LLMTimeoutError(
                    "Request to LLM timed out."
                )
            )

        except APIConnectionError:
            yield StreamError(
                error=LLMConnectionError(
                    "Unable to connect to the LLM provider."
                )
            )

        except AuthenticationError:
            yield StreamError(
                error=LLMAuthenticationError(
                    "Unable to authenticate with the LLM provider."
                )
            )

        except BadRequestError as exc:
            yield StreamError(
                error=LLMBadRequestError(
                    f"Invalid request to LLM provider {exc}"
                )
            )

        except APIStatusError as exc:
            yield StreamError(
                error=LLMBadRequestError(
                    f"LLM provider returned an error: {exc}"
                )
            )

        except (json.JSONDecodeError, ValueError) as exc:
            yield StreamError(
                error=LLMBadResponseError(
                    f"Unable to interpret the streamed response: {exc}"
                )
            )

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from uuid import uuid4

from atlas_ai.errors import (
    AtlasError,
    LLMBadResponseError,
)
from atlas_ai.llm.client import AssistantLLMClient
from atlas_ai.llm.streaming import (
    ResponseCompleted,
    StreamError,
    StreamEvent,
    TextDelta,
)
from atlas_ai.models import (
    AssistantMessage,
    DeveloperMessage,
    TokenUsage,
    UserMessage,
)
from atlas_ai.prompts import PROMPTS
from atlas_ai.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


def _aggregate_usage(
        usages: list[TokenUsage | None]
) -> TokenUsage | None:
    if not usages or any(usage is None for usage in usages):
        return None

    known_usages = [usage for usage in usages if usage is not None]

    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in known_usages),
        output_tokens=sum(u.output_tokens for u in known_usages),
        total_tokens=sum(u.total_tokens for u in known_usages),
        cached_input_tokens=sum(u.cached_input_tokens for u in known_usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in known_usages),
    )


class AssistantService:
    def __init__(
        self,
        llm_client: AssistantLLMClient,
        tool_executor: ToolExecutor
    ):
        """
        Initialize the service layer assistant with
        the LLM client and the tool executor.
        """
        self.client = llm_client
        self.tool_executor = tool_executor
        developer_message = DeveloperMessage(content=PROMPTS["main"])
        self.conversation = [developer_message]

    def add_to_conversation(
        self,
        message: AssistantMessage | DeveloperMessage | UserMessage,
    ) -> None:
        """
        Adds message to the conversation.
        Args:
            - message: Message could be from the user,
            developer or the assistant.
        """
        self.conversation.append(message)

    def generate_response(self, user_input: str) -> str:
        """
        Get responses from AI model. Execute tools if
        model makes tool calls.
        """
        request_id = str(uuid4())

        logger.info("Request started | request_id=%s", request_id)

        user_message = UserMessage(content=user_input)

        self.add_to_conversation(user_message)

        context = self.conversation.copy()

        usages: list[TokenUsage | None] = []

        try:
            while True:
                logger.info(
                    "Preparing request to LLM | request_id=%s",
                    request_id
                )

                response = self.client.generate(
                    context=context,
                )

                usages.append(response.usage)

                assistant_message = response.message
                context.append(assistant_message)

                if not assistant_message.tool_calls:
                    break

                tool_result_message = self.tool_executor.execute(
                    assistant_message.tool_calls
                )

                context.append(tool_result_message)

            self.add_to_conversation(assistant_message)

            total_usage = _aggregate_usage(usages)

            if total_usage is None:
                logger.info(
                    "Interaction usage unavailable | request_id=%s",
                    request_id
                )
            else:
                logger.info(
                    (
                        "Interation usage | request_id=%s "
                        "input_tokens=%d output_tokens=%d total_tokens=%d "
                        "cached_input_tokens=%d reasoning_tokens=%d"
                    ),
                    request_id,
                    total_usage.input_tokens,
                    total_usage.output_tokens,
                    total_usage.total_tokens,
                    total_usage.cached_input_tokens,
                    total_usage.reasoning_tokens,
                )

            logger.info(
                "Request completed | request_id=%s",
                request_id
            )

            return assistant_message.text

        except AtlasError:
            # remove last user input from conversation
            if self.conversation and self.conversation[-1] == user_message:
                self.conversation.pop()

            logger.error(
                "Request failed | request_id=%s",
                request_id
            )
            raise

    def stream_response(self, user_input: str) -> Iterator[StreamEvent]:
        """Stream a response, executing tool calls when necessary."""
        request_id = str(uuid4())

        logger.info("Streaming request started | request_id=%s", request_id)

        user_message = UserMessage(content=user_input)
        self.add_to_conversation(user_message)
        context = self.conversation.copy()

        usages: list[TokenUsage | None] = []

        try:
            while True:
                final_message = None

                for event in self.client.stream(context):
                    if isinstance(event, TextDelta):
                        yield event

                    elif isinstance(event, StreamError):
                        self.conversation.pop()
                        yield event
                        return

                    elif isinstance(event, ResponseCompleted):
                        final_message = event.message
                        usages.append(event.usage)

                if final_message is None:
                    raise LLMBadResponseError(
                        "The stream ended without a completed response."
                    )

                context.append(final_message)

                if not final_message.tool_calls:
                    self.add_to_conversation(final_message)

                    total_usage = _aggregate_usage(usages)

                    if total_usage is None:
                        logger.info(
                            "Interaction usage unavailable | request_id=%s",
                            request_id
                        )
                    else:
                        logger.info(
                            (
                                "Interation usage | request_id=%s "
                                "input_tokens=%d output_tokens=%d "
                                "total_tokens=%d cached_input_tokens=%d "
                                "reasoning_tokens=%d"
                            ),
                            request_id,
                            total_usage.input_tokens,
                            total_usage.output_tokens,
                            total_usage.total_tokens,
                            total_usage.cached_input_tokens,
                            total_usage.reasoning_tokens,
                        )

                    logger.info(
                        "Streaming request completed | request_id=%s",
                        request_id
                    )

                    yield ResponseCompleted(
                        message=final_message,
                        usage=total_usage,
                    )
                    return

                tool_result_message = self.tool_executor.execute(
                    final_message.tool_calls
                )
                context.append(tool_result_message)

        except AtlasError as exc:
            if self.conversation and self.conversation[-1] == user_message:
                self.conversation.pop()

            logger.error(
                "Streaming request failed | request_id=%s",
                request_id
            )
            yield StreamError(error=exc)

    async def astream_response(
        self,
        user_input: str
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronously stream a response and handle tool calls."""
        request_id = str(uuid4())

        logger.info(
            "Async streaming request started | request_id=%s",
            request_id
        )

        user_message = UserMessage(content=user_input)
        self.add_to_conversation(user_message)
        context = self.conversation.copy()

        usages: list[TokenUsage | None] = []

        try:
            while True:
                final_message = None

                async for event in self.client.astream(context):
                    if isinstance(event, TextDelta):
                        yield event

                    elif isinstance(event, StreamError):
                        if (
                            self.conversation
                            and self.conversation[-1] == user_message
                        ):
                            self.conversation.pop()
                        yield event
                        return

                    elif isinstance(event, ResponseCompleted):
                        final_message = event.message
                        usages.append(event.usage)

                if final_message is None:
                    raise LLMBadResponseError(
                        "The async stream ended without a completed response."
                    )

                context.append(final_message)

                if not final_message.tool_calls:
                    self.add_to_conversation(final_message)

                    total_usage = _aggregate_usage(usages)

                    if total_usage is None:
                        logger.info(
                            "Interaction usage unavailable | request_id=%s",
                            request_id
                        )
                    else:
                        logger.info(
                            (
                                "Interation usage | request_id=%s "
                                "input_tokens=%d output_tokens=%d "
                                "total_tokens=%d cached_input_tokens=%d "
                                "reasoning_tokens=%d"
                            ),
                            request_id,
                            total_usage.input_tokens,
                            total_usage.output_tokens,
                            total_usage.total_tokens,
                            total_usage.cached_input_tokens,
                            total_usage.reasoning_tokens,
                        )

                    logger.info(
                        "Async streaming request completed | request_id=%s",
                        request_id
                    )

                    yield ResponseCompleted(
                        message=final_message,
                        usage=total_usage
                    )
                    return

                # Move synchronous ToolExecutor off the event loop.
                tool_result_message = await asyncio.to_thread(
                    self.tool_executor.execute,
                    final_message.tool_calls
                )

                context.append(tool_result_message)

        except AtlasError as exc:
            if self.conversation and self.conversation[-1] == user_message:
                self.conversation.pop()

            logger.error(
                "Async streaming failed | request_id=%s",
                request_id
            )
            yield StreamError(error=exc)

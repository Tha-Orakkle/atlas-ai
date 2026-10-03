import logging
from uuid import uuid4

from atlas_ai.errors import AtlasError
from atlas_ai.llm.client import LLMClient
from atlas_ai.models import (
    AssistantMessage,
    DeveloperMessage,
    UserMessage,
)
from atlas_ai.prompts import PROMPTS
from atlas_ai.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class AssistantService:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor
    ):
        """
        Initialize the service layer assistant with
        the LLM client and the tool executor.
        """
        self.client = llm_client
        self.tool_executor = tool_executor
        self.conversation = [
            DeveloperMessage(
                content=PROMPTS["main"]
            )
        ]

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
        Args:
        - user_input: input by user
        """
        request_id = str(uuid4())

        logger.info("Request started | request_id=%s", request_id)

        self.add_to_conversation(
            UserMessage(
                content=user_input
            ))

        context = self.conversation.copy()

        try:
            while True:
                logger.info(
                    "Preparing request to LLM | request_id=%s",
                    request_id
                )

                response = self.client.generate(
                    context=context,
                )

                assistant_message = response.message
                context.append(assistant_message)

                if not assistant_message.tool_calls:
                    break

                tool_result_message = self.tool_executor.execute(
                    assistant_message.tool_calls
                )

                context.append(tool_result_message)

            self.add_to_conversation(assistant_message)

            logger.info(
                "Request completed | request_id=%s",
                request_id
            )

            return assistant_message.text

        except AtlasError:
            # remove last user input from conversation
            self.conversation.pop()

            logger.error(
                "Request failed | request_id=%s",
                request_id
            )
            raise

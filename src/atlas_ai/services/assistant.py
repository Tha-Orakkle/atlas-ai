import logging

from uuid import uuid4
from atlas_ai.errors import AtlasError
from atlas_ai.llm.client import LLMClient
from atlas_ai.tools.executor import ToolExecutor
from atlas_ai.prompts import PROMPTS

logger = logging.getLogger(__name__)


class AssistantService:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor
    ):
        self.client = llm_client
        self.tool_executor = tool_executor
        self.context = []
        self.add_to_context(
            role="developer",
            content=PROMPTS["main"]
        )

    def add_to_context(self, role: str, content: str) -> None:
        """
        Adds input/response from user/assistant to the conversation.
        Args:
            - role (str): user or assistant.
            - content (str): the actual input by user or response from
              the assistant
        """
        self.context.append({
            "role": role,
            "content": content
        })


    def generate_response(self, user_input: str) -> str:
        """
        Get responses from AI model. Execute tools if
        model makes tool calls.
        Args:
        - user_input: input by user
        """
        request_id = str(uuid4())

        logger.info("Request started | request_id=%s", request_id)

        self.add_to_context("user", user_input)

        input_list = self.context.copy()

        try:
            while True:
                logger.info(
                    "Calling LLM | request_id=%s",
                    request_id
                )

                response = self.client.generate(
                    context=input_list,
                )

                input_list += response.output
                tools_output = self.tool_executor.execute(response.output)

                if not tools_output:
                    break
                input_list += tools_output

            self.add_to_context("assistant", response.output_text)

            logger.info(
                "Request completed | request_id=%s",
                request_id
            )

            return response.output_text

        except AtlasError as exc:
            logger.error(
                "Request failed | request_id=%s",
                request_id
            )
            raise

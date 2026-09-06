import json
import logging

from uuid import uuid4
from atlas_ai.errors import AtlasError
from atlas_ai.tools.registry import TOOLS
from atlas_ai.prompts import PROMPTS

logger = logging.getLogger(__name__)


class AssistantService:
    def __init__(self, llm_client):
        self.client = llm_client
        self.tools_registry = TOOLS
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

    def make_tool_output(self, call_id: str, output: dict) -> dict:
        """
        Make tool output.
        Args:
            - call_id (str): The ID of the tool call from the model.
            - output (dict): result of the tool.
        Returns:
            - a dict to be sent back to the model with the
              type 'function_call_output'.
        """
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output)
        }

    def execute_tools(self, response_output: list) -> list[dict]:
        """"
        Gets and executes the tool called by model.
        Args:
            - response_output: list of responses from the AI model.
        Returns:
            - list of all function_call_outputs
        """
        tools_output = []

        for item in response_output:
            if item.type != "function_call":
                continue

            logger.info(
                "Executing tool | tool=%s",
                item.name
            )

            tool = self.tools_registry.get(item.name)
            if not tool:
                tools_output.append(self.make_tool_output(
                    call_id=item.call_id,
                    output={"error": f"Unknown tool: {item.name}"}
                ))

                logger.exception(
                    "Tool execution failed. Tool not found. | tool=%s",
                    item.name
                )

                continue

            args = json.loads(item.arguments)
            result = tool["function"](**args)
            tools_output.append(self.make_tool_output(
                call_id=item.call_id, output=result
            ))

            logger.info(
                "Tool completed | tool=%s",
                item.name
            )

        return tools_output

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
                tools_output = self.execute_tools(response.output)
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
            logging.exception(
                "Request failed | request_id=%s",
                request_id
            )
            raise

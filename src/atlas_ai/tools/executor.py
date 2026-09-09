import json
import logging

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    @staticmethod
    def _make_tool_output(
        call_id: str,
        output: dict
    ) -> dict:
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

    def execute(self, response_output: list) -> list:
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
            tool = self.tool_registry.get(item.name)

            if not tool:
                logger.error(
                    "Tool not found | tool=%s",
                    item.name
                )
                tools_output.append(
                    self._make_tool_output(
                        call_id=item.call_id,
                        output={"error": f"Unknown tool: {item.name}"}
                    )
                )
                continue

            try:
                args = json.loads(item.arguments)
                result = tool["function"](**args)

            except Exception as exc:
                logger.error(
                    "Tool execution failed | tool=%s",
                    item.name
                )
                result = {"error": "Tool execution failed."}

            tools_output.append(
                self._make_tool_output(
                    call_id=item.call_id,
                    output=result
                )
            )

            logger.info(
                "Tool completed | tool=%s",
                item.name
            )

        return tools_output

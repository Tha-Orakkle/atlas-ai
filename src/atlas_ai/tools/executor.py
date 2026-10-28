import logging
from typing import Any

from atlas_ai.models import ToolCall, ToolResult, ToolResultMessage
from atlas_ai.observability import RequestContext

logger = logging.getLogger(__name__)


class ToolExecutor:

    def __init__(
        self,
        tools_registry: dict[str, Any]
    ):
        """
        Initialize tool executor with the tool registry.
        """
        self.tools_registry = tools_registry

    def _build_tool_result(
        self,
        call_id: str,
        result: dict[str, Any]
    ) -> ToolResult:
        """
        Convert tool result to application-level
        ToolResult object.
        Args:
            - call_id: model tool call ID.
            - result: result of the tool.
        """

        return ToolResult(
            call_id=call_id,
            result=result
        )

    def execute(
        self,
        tool_calls: list[ToolCall],
        *,
        request_context: RequestContext | None = None
    ) -> ToolResultMessage:

        """"
        Gets and executes the tools requested by model.
        Args:
            - tool_calls: list of all ToolCall requests.
        Returns:
            - ToolResultMessage containing all tool results.
        """
        tool_results = []

        for tool_call in tool_calls:
            logger.info(
                "Executing tool | tool=%s interaction_id=%s "
                "request_id=%s",
                tool_call.name,
                request_context.interaction_id if request_context else None,
                request_context.request_id if request_context else None,
            )

            tool = self.tools_registry.get(tool_call.name)

            if not tool:
                logger.error(
                    "Tool not found | tool=%s interaction_id=%s "
                    "request_id=%s",
                    tool_call.name,
                    request_context.interaction_id
                    if request_context else None,
                    request_context.request_id
                    if request_context else None,
                )
                tool_results.append(
                    self._build_tool_result(
                        call_id=tool_call.call_id,
                        result={
                            "error": f"Unknown tool: '{tool_call.name}'"
                        }
                    )
                )
                continue

            try:
                result = tool["function"](**tool_call.arguments)
                tool_results.append(
                    self._build_tool_result(
                        call_id=tool_call.call_id,
                        result=result
                    )
                )

            except Exception as exc:  # update to application level errors  # noqa: BLE001
                # convert to logger.exception
                logger.error(
                    "Tool execution failed | tool=%s interaction_id=%s "
                    "request_id=%s error=%s",
                    tool_call.name,
                    request_context.interaction_id
                    if request_context else None,
                    request_context.request_id
                    if request_context else None,
                    str(exc)
                )
                tool_results.append(
                    self._build_tool_result(
                        call_id=tool_call.call_id,
                        result={
                            "error": f"Tool failed: {exc}."
                        }
                    )
                )
                continue

            logger.info(
                "Tool execution completed | tool=%s "
                "interaction_id=%s request_id=%s",
                tool_call.name,
                request_context.interaction_id if request_context else None,
                request_context.request_id if request_context else None
            )

        return ToolResultMessage(results=tool_results)

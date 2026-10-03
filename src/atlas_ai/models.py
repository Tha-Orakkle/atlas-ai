from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    result: Any


@dataclass(frozen=True)
class AssistantMessage:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True)
class DeveloperMessage:
    content: str


@dataclass(frozen=True)
class ToolResultMessage:
    results: list[ToolResult]


@dataclass(frozen=True)
class UserMessage:
    content: str


ConversationMessage = (
    DeveloperMessage
    | UserMessage
    | AssistantMessage
    | ToolResultMessage
)


@dataclass(frozen=True)
class LLMResponse:
    message: AssistantMessage

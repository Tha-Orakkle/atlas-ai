from dataclasses import dataclass

from atlas_ai.models import AssistantMessage, TokenUsage, ToolCall


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallCompleted:
    tool_call: ToolCall


@dataclass(frozen=True)
class ResponseCompleted:
    message: AssistantMessage
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class StreamError:
    error: Exception


StreamEvent = (
    TextDelta
    | ToolCallCompleted
    | ResponseCompleted
    | StreamError
)

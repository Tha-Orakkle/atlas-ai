class AtlasError(Exception):
    """Base exception for Atlas"""

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None
    ):
        super().__init__(message)
        self.retry_after = retry_after


class LLMError(AtlasError):
    """Base exception for LLM-related errors."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits Atlas."""

    retryable = True


class LLMAuthenticationError(LLMError):
    """Raised when LLM authentication fails."""

    retryable = False


class LLMTimeoutError(LLMError):
    """Raised when the LLM request times out."""

    retryable = True


class LLMConnectionError(LLMError):
    """Raised when Atlas cannot connect to the LLM provider."""

    retryable = True


class LLMServerError(LLMError):
    """The LLM provider returned a temporary server error."""

    retryable = True


class LLMBadRequestError(LLMError):
    """Raised when the LLM retuns a bad request error."""

    retryable = False


class LLMResponseError(LLMError):
    """Raised when the LLM returns an unusable response."""


# ====================================================
# TOOL ERRORS
# ====================================================
class ToolError(AtlasError):
    """Base exception for tool-related errors."""


class UnknownToolError(ToolError):
    """Raised when the model requests an unavailable tool."""


class ToolExecutionError(ToolError):
    """Raised when execution of a tool fails."""

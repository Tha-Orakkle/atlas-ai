class AtlasError(Exception):
    """Base exception for Atlas"""


class LLMError(AtlasError):
    """Base exception for LLM-related errors."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits Atlas."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM authentication fails."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM request times out."""


class LLMConnectionError(LLMError):
    """Raised when Atlas cannot connect to the LLM provider."""


class LLMResponseError(LLMError):
    """Raised when the LLM returns an unusable response."""


class LLMBadRequestError(LLMError):
    """Raised when the LLM retuns a bad request error."""


# ====================================================
# TOOL ERRORS
# ====================================================
class ToolError(AtlasError):
    """Base exception for tool-related errors."""


class UnknownToolError(ToolError):
    """Raised when the model requests an unavailable tool."""


class ToolExecutionError(ToolError):
    """Raised when execution of a tool fails."""

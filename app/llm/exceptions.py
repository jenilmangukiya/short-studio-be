"""Custom exceptions for the LLM layer."""


class LLMError(Exception):
    """Base exception for all LLM errors."""


class LLMConfigError(LLMError):
    """Raised when configuration is invalid or missing (e.g. unknown task, missing API key)."""


class LLMRateLimitError(LLMError):
    """Raised when an LLM provider returns a rate-limit (429) error."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout."""


class LLMProviderError(LLMError):
    """Raised when an LLM provider fails (e.g. server error, validation failure after retry)."""


class LLMTruncatedError(LLMProviderError):
    """Raised when model generation is cut off due to reaching the maximum token limit."""


"""Swappable LLM layer for Shorts Studio."""

from .base import BaseLLM
from .exceptions import (
    LLMConfigError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTruncatedError,
)
from .router import (
    ALLOWED_TASKS,
    create_llm_instance,
    get_llm,
    get_routing_table,
    resolve_task_routing,
)
from .types import LLMResult

__all__ = [
    "BaseLLM",
    "LLMResult",
    "LLMError",
    "LLMConfigError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMProviderError",
    "LLMTruncatedError",
    "ALLOWED_TASKS",
    "get_llm",
    "create_llm_instance",
    "get_routing_table",
    "resolve_task_routing",
]

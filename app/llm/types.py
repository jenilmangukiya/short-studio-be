"""Types and data structures for the LLM layer."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResult:
    """Standardized result object returned by all LLM calls.

    Attributes:
        output: Text string or validated Pydantic model instance.
        provider: Name of the LLM provider (e.g. 'mock', 'gemini', 'openai').
        model: Specific model name used for the generation.
        input_tokens: Prompt/input token count.
        output_tokens: Completion/candidate output token count.
        latency_seconds: Total execution latency in seconds.
        task: Optional pipeline task name (e.g. 'script', 'storyboard').
        thinking_tokens: Thinking / reasoning tokens if reported by the provider.
        finish_reason: Finish reason reported by the provider (e.g. 'stop', 'length').
    """

    output: Any
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    task: str | None = None
    thinking_tokens: int | None = None
    finish_reason: str | None = None

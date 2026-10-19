"""Base interface and common logic for LLM providers."""

import time
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import (
    LLMConfigError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTruncatedError,
)
from .types import LLMResult

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The identifier of this provider (e.g. 'mock', 'gemini', 'openai')."""
        pass

    def _check_truncation(self, finish: str | None) -> None:
        """Raise LLMTruncatedError if the model finish reason indicates truncation."""
        if finish and finish.upper() in ("MAX_TOKENS", "LENGTH"):
            raise LLMTruncatedError(
                f"{self.provider_name} output hit the token limit and was cut off (finish_reason: '{finish}'). "
                f"Consider raising LLM_MAX_OUTPUT_TOKENS (currently set to {self.max_output_tokens})."
            )

    @abstractmethod
    def _execute_text(self, prompt: str, system_prompt: str | None) -> tuple[str, int, int, int | None, str | None]:
        """Provider-specific text generation implementation.

        Returns:
            Tuple of (text_output, input_tokens, output_tokens, thinking_tokens, finish_reason)
        """
        pass

    @abstractmethod
    def _execute_structured(
        self, prompt: str, response_model: Type[T], system_prompt: str | None
    ) -> tuple[Any, int, int, int | None, str | None]:
        """Provider-specific native structured output generation.

        Returns:
            Tuple of (raw_or_parsed_output, input_tokens, output_tokens, thinking_tokens, finish_reason)
        """
        pass

    def generate(self, prompt: str, system_prompt: str | None = None, task: str | None = None) -> LLMResult:
        """Generate text from a prompt, optionally including a system prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            task: Pipeline task name for logging and tracking.

        Returns:
            LLMResult containing the generated text and call metadata.
        """
        start = time.perf_counter()
        output, in_tokens, out_tokens, think_tokens, finish = self._execute_text(prompt, system_prompt)
        latency = time.perf_counter() - start

        self._check_truncation(finish)

        return LLMResult(
            output=output,
            provider=self.provider_name,
            model=self.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            latency_seconds=round(latency, 4),
            task=task,
            thinking_tokens=think_tokens,
            finish_reason=finish,
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str | None = None,
        task: str | None = None,
    ) -> LLMResult:
        """Generate structured output validated against a Pydantic model.

        Uses native provider structured output. If validation fails, retries once
        with the validation error included in the prompt.
        """
        start = time.perf_counter()

        # Attempt 1
        raw, in_tokens, out_tokens, think_tokens, finish = self._execute_structured(
            prompt, response_model, system_prompt
        )
        self._check_truncation(finish)

        validated, err = self._validate_model(raw, response_model)

        if validated is not None:
            latency = time.perf_counter() - start
            return LLMResult(
                output=validated,
                provider=self.provider_name,
                model=self.model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_seconds=round(latency, 4),
                task=task,
                thinking_tokens=think_tokens,
                finish_reason=finish,
            )

        # Attempt 2 (Retry once with error message)
        retry_prompt = (
            f"{prompt}\n\n"
            f"[IMPORTANT: The previous output failed schema validation with error: {err}]. "
            f"Please return a valid JSON object strictly adhering to the schema."
        )

        raw_retry, in_tokens_2, out_tokens_2, think_tokens_2, finish_2 = self._execute_structured(
            retry_prompt, response_model, system_prompt
        )
        self._check_truncation(finish_2)
        validated_retry, err_retry = self._validate_model(raw_retry, response_model)

        total_in = in_tokens + in_tokens_2
        total_out = out_tokens + out_tokens_2
        total_think = None
        if think_tokens is not None or think_tokens_2 is not None:
            total_think = (think_tokens or 0) + (think_tokens_2 or 0)

        latency = time.perf_counter() - start

        if validated_retry is not None:
            return LLMResult(
                output=validated_retry,
                provider=self.provider_name,
                model=self.model,
                input_tokens=total_in,
                output_tokens=total_out,
                latency_seconds=round(latency, 4),
                task=task,
                thinking_tokens=total_think,
                finish_reason=finish_2,
            )

        raise LLMProviderError(
            f"Structured output failed validation for {response_model.__name__} after 1 retry: {err_retry}"
        )

    def _validate_model(self, data: Any, response_model: Type[T]) -> tuple[T | None, str | None]:
        """Validate raw string or dictionary data against response_model."""
        if isinstance(data, response_model):
            return data, None
        try:
            if isinstance(data, str):
                return response_model.model_validate_json(data), None
            elif isinstance(data, dict):
                return response_model.model_validate(data), None
            elif hasattr(data, "model_dump"):
                return response_model.model_validate(data.model_dump()), None
            return response_model.model_validate(data), None
        except (ValidationError, Exception) as exc:
            return None, str(exc)

"""OpenAI LLM provider using the official openai SDK."""

import time
from typing import Any, Type, TypeVar
from pydantic import BaseModel

from .base import BaseLLM
from .exceptions import (
    LLMConfigError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

T = TypeVar("T", bound=BaseModel)


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation supporting native structured outputs and reasoning controls."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazily initialize OpenAI client to avoid import-time key errors."""
        if self._client is None:
            if not self.api_key:
                raise LLMConfigError(
                    "Missing OPENAI_API_KEY. Please set OPENAI_API_KEY in your .env file or environment."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                raise LLMConfigError(f"Failed to initialize OpenAI client: {exc}") from None
        return self._client

    def _build_kwargs(self, is_reasoning_model: bool = False) -> dict[str, Any]:
        """Construct common arguments for OpenAI chat completions."""
        kwargs: dict[str, Any] = {
            "max_completion_tokens": self.max_output_tokens,
        }

        # Reasoning models (like o1, o3-mini) do not accept temperature, but support reasoning_effort
        if self.reasoning_effort or is_reasoning_model:
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature

        return kwargs

    def _call_with_retry(self, action_desc: str, func):
        """Execute OpenAI call with up to 2 retries on rate limits (429) or transient 5xx errors."""
        import openai

        max_retries = 2
        backoff = 1.0

        for attempt in range(max_retries + 1):
            try:
                return func()
            except openai.AuthenticationError as exc:
                raise LLMConfigError(f"OpenAI authentication failed: {exc.message}") from None
            except openai.BadRequestError as exc:
                raise LLMProviderError(f"OpenAI invalid request: {exc.message}") from None
            except openai.RateLimitError as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise LLMRateLimitError(f"OpenAI rate limit exceeded: {exc.message}") from None
            except (openai.APITimeoutError, TimeoutError) as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise LLMTimeoutError(f"OpenAI request timed out after {self.timeout_seconds}s: {exc}") from None
            except (openai.InternalServerError, openai.APIConnectionError) as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise LLMProviderError(f"OpenAI server error ({action_desc}): {exc}") from None
            except openai.APIError as exc:
                raise LLMProviderError(f"OpenAI API error ({action_desc}): {exc.message}") from None
            except Exception as exc:
                if isinstance(exc, (LLMConfigError, LLMRateLimitError, LLMTimeoutError, LLMProviderError)):
                    raise
                raise LLMProviderError(f"Unexpected OpenAI error ({action_desc}): {exc}") from None

    def _extract_metadata(self, completion) -> tuple[int, int, int | None, str | None]:
        """Extract prompt tokens, completion tokens, reasoning tokens, and finish reason."""
        in_tokens = 0
        out_tokens = 0
        thinking_tokens = None
        finish_reason = None

        if hasattr(completion, "usage") and completion.usage:
            in_tokens = completion.usage.prompt_tokens or 0
            out_tokens = completion.usage.completion_tokens or 0
            details = getattr(completion.usage, "completion_tokens_details", None)
            if details:
                thinking_tokens = getattr(details, "reasoning_tokens", None)

        if completion.choices:
            finish_reason = completion.choices[0].finish_reason

        return in_tokens, out_tokens, thinking_tokens, finish_reason

    def _execute_text(
        self, prompt: str, system_prompt: str | None
    ) -> tuple[str, int, int, int | None, str | None]:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        is_o_model = self.model.startswith("o1") or self.model.startswith("o3")
        kwargs = self._build_kwargs(is_reasoning_model=is_o_model)

        completion = self._call_with_retry(
            "generate text",
            lambda: client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            ),
        )

        choice = completion.choices[0]
        text = choice.message.content or ""
        in_tokens, out_tokens, thinking_tokens, finish = self._extract_metadata(completion)
        return text, in_tokens, out_tokens, thinking_tokens, finish

    def _execute_structured(
        self, prompt: str, response_model: Type[T], system_prompt: str | None
    ) -> tuple[Any, int, int, int | None, str | None]:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        is_o_model = self.model.startswith("o1") or self.model.startswith("o3")
        kwargs = self._build_kwargs(is_reasoning_model=is_o_model)

        completion = self._call_with_retry(
            f"generate structured {response_model.__name__}",
            lambda: client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
                **kwargs,
            ),
        )

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMProviderError(f"OpenAI refused structured output: {message.refusal}")

        parsed_or_raw = getattr(message, "parsed", None) or message.content or ""
        in_tokens, out_tokens, thinking_tokens, finish = self._extract_metadata(completion)
        return parsed_or_raw, in_tokens, out_tokens, thinking_tokens, finish

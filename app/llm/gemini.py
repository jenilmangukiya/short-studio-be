"""Google Gemini LLM provider using the official google-genai SDK."""

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


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation with configurable thinking level."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking_level: str = "minimal",
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
        self.thinking_level = thinking_level
        self._client = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self):
        """Lazily initialize Google GenAI client to avoid import-time key errors."""
        if self._client is None:
            if not self.api_key:
                raise LLMConfigError(
                    "Missing GEMINI_API_KEY. Please set GEMINI_API_KEY in your .env file or environment."
                )
            try:
                from google import genai
                # http_options timeout is in milliseconds in google-genai
                timeout_ms = int(self.timeout_seconds * 1000)
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options={"timeout": timeout_ms},
                )
            except Exception as exc:
                raise LLMConfigError(f"Failed to initialize Gemini client: {exc}") from None
        return self._client

    def _build_config(
        self,
        system_prompt: str | None = None,
        response_model: Type[T] | None = None,
    ):
        """Construct GenerateContentConfig with thinking and structured output settings."""
        from google.genai import types

        thinking_map = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        level = thinking_map.get(self.thinking_level.lower(), types.ThinkingLevel.MINIMAL)
        thinking_config = types.ThinkingConfig(thinking_level=level)

        kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "thinking_config": thinking_config,
        }

        if system_prompt:
            kwargs["system_instruction"] = system_prompt

        if response_model is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_model

        return types.GenerateContentConfig(**kwargs)

    def _call_with_retry(self, action_desc: str, func):
        """Execute Gemini API call with up to 2 retries on rate limits (429) or 5xx errors."""
        from google.genai.errors import APIError

        max_retries = 2
        backoff = 1.0

        for attempt in range(max_retries + 1):
            try:
                return func()
            except APIError as exc:
                code = getattr(exc, "code", None)
                msg = getattr(exc, "message", str(exc))

                # Auth errors (401, 403) - do not retry
                if code in (401, 403):
                    raise LLMConfigError(f"Gemini authentication failed (HTTP {code}): {msg}") from None

                # Bad request (400) - do not retry
                if code == 400:
                    raise LLMProviderError(f"Gemini invalid request (HTTP 400): {msg}") from None

                # Rate limit (429) - retry
                if code == 429:
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LLMRateLimitError(f"Gemini rate limit exceeded: {msg}") from None

                # Server error (5xx) - retry
                if code and code >= 500:
                    if attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise LLMProviderError(f"Gemini server error (HTTP {code}): {msg}") from None

                # Other API errors
                raise LLMProviderError(f"Gemini API error ({action_desc}): {msg}") from None
            except TimeoutError as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise LLMTimeoutError(f"Gemini request timed out after {self.timeout_seconds}s: {exc}") from None
            except Exception as exc:
                # If it's already one of our custom errors, re-raise
                if isinstance(exc, (LLMConfigError, LLMRateLimitError, LLMTimeoutError, LLMProviderError)):
                    raise
                raise LLMProviderError(f"Unexpected Gemini error ({action_desc}): {exc}") from None

    def _extract_metadata(self, response) -> tuple[int, int, int | None, str | None]:
        """Extract input, output, thinking tokens and finish reason."""
        in_tokens = 0
        out_tokens = 0
        think_tokens = None
        finish_reason = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            think_tokens = getattr(response.usage_metadata, "thoughts_token_count", None)

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason") and candidate.finish_reason is not None:
                finish_reason = str(getattr(candidate.finish_reason, "name", candidate.finish_reason))

        return in_tokens, out_tokens, think_tokens, finish_reason

    def _execute_text(
        self, prompt: str, system_prompt: str | None
    ) -> tuple[str, int, int, int | None, str | None]:
        client = self._get_client()
        config = self._build_config(system_prompt=system_prompt)

        response = self._call_with_retry(
            "generate text",
            lambda: client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            ),
        )

        text = response.text or ""
        in_tokens, out_tokens, think_tokens, finish = self._extract_metadata(response)
        return text, in_tokens, out_tokens, think_tokens, finish

    def _execute_structured(
        self, prompt: str, response_model: Type[T], system_prompt: str | None
    ) -> tuple[Any, int, int, int | None, str | None]:
        client = self._get_client()
        config = self._build_config(system_prompt=system_prompt, response_model=response_model)

        response = self._call_with_retry(
            f"generate structured {response_model.__name__}",
            lambda: client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            ),
        )

        raw_text = response.text or ""
        in_tokens, out_tokens, think_tokens, finish = self._extract_metadata(response)
        return raw_text, in_tokens, out_tokens, think_tokens, finish

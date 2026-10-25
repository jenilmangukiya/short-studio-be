"""OpenAI image provider using the official openai SDK."""

import base64
import time
from typing import Any
import httpx

from .base import BaseImageProvider
from .exceptions import (
    ImageConfigError,
    ImageContentBlockedError,
    ImageError,
    ImageProviderError,
    ImageRateLimitError,
    ImageTimeoutError,
)


class OpenAIImageProvider(BaseImageProvider):
    """OpenAI Image generation provider using client.images.generate."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        est_cost_usd: float | None = None,
    ):
        super().__init__(
            model=model or "",
            timeout_seconds=timeout_seconds,
            est_cost_usd=est_cost_usd,
        )
        self.api_key = api_key
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazily initialize OpenAI client with strict key and model separation."""
        if self._client is None:
            if not self.api_key:
                raise ImageConfigError(
                    "Missing OPENAI_IMAGE_API_KEY. Set OPENAI_IMAGE_API_KEY in your .env file or environment."
                )
            if not self.model or not self.model.strip():
                raise ImageConfigError(
                    "Missing OPENAI_IMAGE_MODEL. Set OPENAI_IMAGE_MODEL in your .env file or environment."
                )
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:
                raise ImageConfigError(f"Failed to initialize OpenAI client: {exc}") from None
        return self._client

    def _is_content_policy_violation(self, exc: Any) -> bool:
        """Check whether an exception or its payload indicates a content-policy violation."""
        code = getattr(exc, "code", None)
        if code == "content_policy_violation":
            return True
        msg = (getattr(exc, "message", None) or str(exc)).lower()
        if "content policy" in msg or "safety system" in msg or "safety violation" in msg:
            return True
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err_dict = body.get("error", {})
            if isinstance(err_dict, dict):
                err_code = err_dict.get("code")
                if err_code == "content_policy_violation":
                    return True
                err_msg = str(err_dict.get("message", "")).lower()
                if "content policy" in err_msg or "safety system" in err_msg:
                    return True
        return False

    def _call_with_retry(self, action_desc: str, func):
        """Execute OpenAI image call with up to 2 retries on rate limits (429), timeouts, and 5xx errors."""
        import openai

        max_retries = 2
        backoff = 1.0

        for attempt in range(max_retries + 1):
            try:
                return func()
            except openai.AuthenticationError as exc:
                raise ImageConfigError(
                    f"OpenAI image authentication failed: {exc.message}. Check OPENAI_IMAGE_API_KEY."
                ) from None
            except openai.PermissionDeniedError as exc:
                raise ImageConfigError(
                    f"OpenAI image permission denied / organization verification required: {exc.message}. "
                    "Verify your organization at https://platform.openai.com/settings/organization/general."
                ) from None
            except openai.BadRequestError as exc:
                if self._is_content_policy_violation(exc):
                    raise ImageContentBlockedError(
                        f"Image prompt rejected by OpenAI safety policy: {exc.message}"
                    ) from None
                raise ImageProviderError(f"OpenAI invalid image request: {exc.message}") from None
            except openai.RateLimitError as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise ImageRateLimitError(
                    f"OpenAI image rate limit exceeded (429): {exc.message}. Please rerun later to resume."
                ) from None
            except (openai.APITimeoutError, TimeoutError, httpx.TimeoutException) as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise ImageTimeoutError(
                    f"OpenAI image request timed out after {self.timeout_seconds}s: {exc}"
                ) from None
            except (openai.InternalServerError, openai.APIConnectionError) as exc:
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise ImageProviderError(
                    f"OpenAI server error ({action_desc}): {exc}"
                ) from None
            except ImageError:
                raise
            except Exception as exc:
                raise ImageProviderError(
                    f"Unexpected OpenAI error ({action_desc}): {exc}"
                ) from None

    def _execute_generate(
        self, prompt: str, size: str, quality: str
    ) -> tuple[bytes, int | None, int | None, int | None]:
        client = self._get_client()

        # Pass size and quality through exactly as configured; do NOT send response_format
        response = self._call_with_retry(
            "generate image",
            lambda: client.images.generate(
                prompt=prompt,
                model=self.model,
                size=size,
                quality=quality,
            ),
        )

        if not response.data:
            raise ImageProviderError("OpenAI returned no image data in response.")

        first_data = response.data[0]

        # Read b64_json if present in the response, otherwise download from url
        if getattr(first_data, "b64_json", None):
            image_bytes = base64.b64decode(first_data.b64_json)
        elif getattr(first_data, "url", None):
            try:
                resp = httpx.get(first_data.url, timeout=self.timeout_seconds)
                resp.raise_for_status()
                image_bytes = resp.content
            except Exception as exc:
                raise ImageProviderError(f"Failed to download image from OpenAI url: {exc}") from exc
        else:
            raise ImageProviderError("OpenAI image response contains neither b64_json nor url.")

        # Extract token usage if provided
        in_tokens = None
        out_tokens = None
        tot_tokens = None
        usage = getattr(response, "usage", None)
        if usage:
            in_tokens = getattr(usage, "input_tokens", None)
            out_tokens = getattr(usage, "output_tokens", None)
            tot_tokens = getattr(usage, "total_tokens", None)

        return image_bytes, in_tokens, out_tokens, tot_tokens

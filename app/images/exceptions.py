"""Custom exceptions for the image provider layer."""


class ImageError(Exception):
    """Base exception for all image generation errors."""


class ImageConfigError(ImageError):
    """Raised when configuration or authentication is invalid or missing
    (e.g. missing API key, missing model name, 401 unauthorized, or 403 organization not verified).
    """


class ImageRateLimitError(ImageError):
    """Raised when an image provider returns a 429 rate limit error after retries are exhausted."""


class ImageTimeoutError(ImageError):
    """Raised when an image request exceeds the configured timeout."""


class ImageContentBlockedError(ImageError):
    """Raised when an image prompt is rejected by safety filters or content moderation policies."""


class ImageProviderError(ImageError):
    """Raised when an image provider encounters an unexpected failure or server error."""

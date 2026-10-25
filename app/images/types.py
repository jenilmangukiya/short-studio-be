"""Types and data structures for the image provider layer."""

from dataclasses import dataclass


@dataclass
class ImageResult:
    """Standardized result returned by all image providers.

    Attributes:
        image_bytes: Raw bytes of the generated image (PNG format).
        provider: Identifier of the provider (e.g. 'mock', 'openai').
        model: Specific model name used.
        size: Image dimensions string (e.g. '1024x1536').
        quality: Image quality setting (e.g. 'low', 'medium', 'high').
        latency_seconds: Time taken to generate the image.
        input_tokens: Input/prompt tokens if reported by the provider.
        output_tokens: Output/completion tokens if reported by the provider.
        total_tokens: Total tokens if reported by the provider.
        est_cost_usd: Estimated cost in USD if configured, or None.
    """

    image_bytes: bytes
    provider: str
    model: str
    size: str
    quality: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    est_cost_usd: float | None = None

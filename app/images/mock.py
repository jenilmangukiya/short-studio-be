"""Mock image provider for local testing and development with zero AI cost."""

import hashlib
import struct
import zlib

from .base import BaseImageProvider


def create_solid_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Generate valid PNG bytes with a solid RGB color using only struct and zlib."""
    # 8-byte PNG file signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk: 13 bytes (width, height, bit_depth=8, color_type=2 [RGB], comp=0, filter=0, interlace=0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT chunk: filter byte (0) + RGB row repeated for each scanline
    raw_row = b"\x00" + bytes([r, g, b]) * width
    raw_data = raw_row * height
    compressed_data = zlib.compress(raw_data, level=6)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed_data) & 0xFFFFFFFF)
    idat_chunk = struct.pack(">I", len(compressed_data)) + b"IDAT" + compressed_data + idat_crc

    # IEND chunk: 0 bytes data
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    return signature + ihdr_chunk + idat_chunk + iend_chunk


class MockImageProvider(BaseImageProvider):
    """Mock image provider requiring no API key and performing no network requests."""

    def __init__(
        self,
        model: str = "mock-image-model",
        timeout_seconds: float = 120.0,
        est_cost_usd: float | None = None,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
            est_cost_usd=est_cost_usd,
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    def _parse_dimensions(self, size: str) -> tuple[int, int]:
        """Extract width and height from size string (e.g. '1024x1536')."""
        try:
            parts = size.strip().lower().split("x")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    return w, h
        except (ValueError, AttributeError):
            pass
        return 1024, 1536

    def _execute_generate(
        self, prompt: str, size: str, quality: str
    ) -> tuple[bytes, int | None, int | None, int | None]:
        # Derive deterministic solid RGB color from SHA256 hash of the prompt
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        r, g, b = digest[0], digest[1], digest[2]

        width, height = self._parse_dimensions(size)
        png_bytes = create_solid_png(width, height, r, g, b)

        in_tokens = max(1, len(prompt.split()))
        out_tokens = 0
        tot_tokens = in_tokens

        return png_bytes, in_tokens, out_tokens, tot_tokens

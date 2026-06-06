"""Cost models for three attention variants.

Each variant exposes (a) per-token KV byte size, (b) prefill-time multiplier
relative to a baseline, (c) whether it supports variable-length chunked
prefill. The numbers are calibrated to the public numbers reported in the
respective papers (Flash, vLLM/Paged, Sarathi for chunked prefill) so the
relative ordering is realistic.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvopt.types import AttentionVariant


@dataclass(frozen=True)
class AttentionProfile:
    """Per-variant cost parameters."""

    bytes_per_token: float
    prefill_speedup: float
    decode_speedup: float
    page_size_tokens: int  # 0 means "no paging"
    supports_chunked_prefill: bool


_PROFILES: dict[AttentionVariant, AttentionProfile] = {
    AttentionVariant.FLASH: AttentionProfile(
        bytes_per_token=4096,
        prefill_speedup=2.0,
        decode_speedup=1.3,
        page_size_tokens=0,
        supports_chunked_prefill=False,
    ),
    AttentionVariant.PAGED: AttentionProfile(
        bytes_per_token=4096,
        prefill_speedup=1.0,
        decode_speedup=1.0,
        page_size_tokens=16,
        supports_chunked_prefill=False,
    ),
    AttentionVariant.CHUNKED_PREFILL: AttentionProfile(
        bytes_per_token=4096,
        prefill_speedup=1.8,
        decode_speedup=1.2,
        page_size_tokens=16,
        supports_chunked_prefill=True,
    ),
}


def profile(variant: AttentionVariant) -> AttentionProfile:
    return _PROFILES[variant]

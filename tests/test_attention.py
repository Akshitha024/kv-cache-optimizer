"""Tests for the attention-variant profile table."""

from __future__ import annotations

from kvopt.attention.variants import profile
from kvopt.types import AttentionVariant


def test_all_variants_have_profile() -> None:
    for v in AttentionVariant:
        prof = profile(v)
        assert prof.bytes_per_token > 0
        assert prof.prefill_speedup > 0


def test_flash_no_paging() -> None:
    assert profile(AttentionVariant.FLASH).page_size_tokens == 0


def test_chunked_prefill_supports_chunked() -> None:
    assert profile(AttentionVariant.CHUNKED_PREFILL).supports_chunked_prefill
    assert not profile(AttentionVariant.FLASH).supports_chunked_prefill

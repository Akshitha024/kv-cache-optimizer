"""Request, attention variant, eviction policy, and run-result types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AttentionVariant(StrEnum):
    FLASH = "flash"
    PAGED = "paged"
    CHUNKED_PREFILL = "chunked_prefill"


class EvictionPolicy(StrEnum):
    LRU = "lru"
    LFU = "lfu"
    TIME = "time"


class Request(BaseModel):
    """A single inference request with prompt and target output lengths."""

    id: int
    arrival_step: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., ge=1)
    output_tokens: int = Field(..., ge=1)


class RunResult(BaseModel):
    """Aggregate outcome of one simulator run."""

    variant: AttentionVariant
    eviction: EvictionPolicy
    n_requests: int
    completed: int
    evicted: int
    peak_kv_mb: float
    throughput_tps: float
    p50_latency_steps: float
    p99_latency_steps: float

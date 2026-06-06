"""Tests for the workload generator."""

from __future__ import annotations

from kvopt.workload.generator import mixed_workload, uniform_workload


def test_mixed_workload_count() -> None:
    reqs = mixed_workload(n=50, seed=7)
    assert len(reqs) == 50


def test_mixed_workload_seed_determinism() -> None:
    a = mixed_workload(n=30, seed=11)
    b = mixed_workload(n=30, seed=11)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]


def test_uniform_workload_constant_lengths() -> None:
    reqs = uniform_workload(n=10, prompt_tokens=256, output_tokens=32)
    assert all(r.prompt_tokens == 256 for r in reqs)
    assert all(r.output_tokens == 32 for r in reqs)


def test_mixed_workload_has_long_tail() -> None:
    reqs = mixed_workload(n=200, seed=17)
    long = [r for r in reqs if r.prompt_tokens >= 4000]
    assert len(long) >= 30  # roughly 30% by construction

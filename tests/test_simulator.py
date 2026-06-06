"""Tests for the simulator + end-to-end runner."""

from __future__ import annotations

from pathlib import Path

from kvopt.runner import sweep
from kvopt.sim.simulator import simulate
from kvopt.types import AttentionVariant, EvictionPolicy
from kvopt.workload.generator import uniform_workload


def test_simulate_completes_small_workload() -> None:
    reqs = uniform_workload(n=10, prompt_tokens=64, output_tokens=4)
    result = simulate(reqs, AttentionVariant.PAGED, EvictionPolicy.LRU, capacity_mb=256)
    assert result.completed == 10
    assert result.evicted == 0


def test_simulate_evicts_when_oversized() -> None:
    reqs = uniform_workload(n=10, prompt_tokens=128_000, output_tokens=4)
    result = simulate(reqs, AttentionVariant.PAGED, EvictionPolicy.LRU, capacity_mb=64)
    assert result.evicted > 0 or result.completed < 10


def test_runner_writes_summary(tmp_path: Path) -> None:
    out = tmp_path / "out"
    s = sweep(out, n_requests=40, seed=1)
    assert s["n_runs"] == 27  # 3 variants x 3 evictions x 3 capacities
    assert (out / "summary.json").exists()

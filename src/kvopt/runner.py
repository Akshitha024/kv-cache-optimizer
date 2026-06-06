"""End-to-end runner: sweep variants x evictions x capacities."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from kvopt.sim.simulator import simulate
from kvopt.types import AttentionVariant, EvictionPolicy, RunResult
from kvopt.viz.charts import (
    eviction_count_bar,
    latency_pareto,
    latency_violin,
    memory_box,
    throughput_grouped_bar,
)
from kvopt.workload.generator import mixed_workload


def sweep(out_dir: Path, n_requests: int = 200, seed: int = 17) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures")

    reqs = mixed_workload(n=n_requests, seed=seed)
    results: list[RunResult] = []
    for variant, evict, cap in itertools.product(
        list(AttentionVariant), list(EvictionPolicy), [64, 128, 256]
    ):
        results.append(simulate(reqs, variant, evict, capacity_mb=cap))

    throughput_grouped_bar(results, figs / "throughput.png")
    memory_box(results, figs / "memory.png")
    latency_pareto(results, figs / "latency_pareto.png")
    eviction_count_bar(results, figs / "evictions.png")
    latency_violin(results, figs / "latency_violin.png")

    summary = {
        "n_runs": len(results),
        "n_requests": n_requests,
        "results": [r.model_dump() for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary

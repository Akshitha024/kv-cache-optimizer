"""Five chart families for the KV-cache benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kvopt.types import RunResult


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def throughput_grouped_bar(rows: list[RunResult], out: Path) -> Path:
    variants = sorted({r.variant.value for r in rows})
    evictions = sorted({r.eviction.value for r in rows})
    x = np.arange(len(variants))
    w = 0.8 / max(1, len(evictions))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, e in enumerate(evictions):
        ys = [
            next(
                (r.throughput_tps for r in rows if r.variant.value == v and r.eviction.value == e),
                0.0,
            )
            for v in variants
        ]
        ax.bar(x + i * w - w * (len(evictions) - 1) / 2, ys, w, label=e)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("tokens / step")
    ax.set_title("Throughput by attention variant and eviction")
    ax.legend()
    return _save(fig, out)


def memory_box(rows: list[RunResult], out: Path) -> Path:
    by_variant: dict[str, list[float]] = {}
    for r in rows:
        by_variant.setdefault(r.variant.value, []).append(r.peak_kv_mb)
    keys = sorted(by_variant)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([by_variant[k] for k in keys], tick_labels=keys)
    ax.set_ylabel("peak KV (MB)")
    ax.set_title("Peak KV memory by attention variant")
    return _save(fig, out)


def latency_pareto(rows: list[RunResult], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = plt.get_cmap("tab10")
    for i, v in enumerate(sorted({r.variant.value for r in rows})):
        xs = [r.throughput_tps for r in rows if r.variant.value == v]
        ys = [r.p99_latency_steps for r in rows if r.variant.value == v]
        ax.scatter(xs, ys, label=v, color=palette(i), alpha=0.8)
    ax.set_xlabel("throughput (tokens/step)")
    ax.set_ylabel("p99 latency (steps)")
    ax.set_title("Throughput vs p99 latency Pareto")
    ax.legend()
    return _save(fig, out)


def eviction_count_bar(rows: list[RunResult], out: Path) -> Path:
    evictions = sorted({r.eviction.value for r in rows})
    variants = sorted({r.variant.value for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(len(variants))
    for i, e in enumerate(evictions):
        ys = [
            sum(r.evicted for r in rows if r.variant.value == v and r.eviction.value == e)
            for v in variants
        ]
        ax.bar(x + i * width - width, ys, width, label=e)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("evicted sessions")
    ax.set_title("Eviction count by variant + policy")
    ax.legend()
    return _save(fig, out)


def latency_violin(rows: list[RunResult], out: Path) -> Path:
    variants = sorted({r.variant.value for r in rows})
    data = []
    for v in variants:
        vals = []
        for r in rows:
            if r.variant.value == v:
                vals.extend([r.p50_latency_steps, r.p99_latency_steps])
        data.append(vals or [0.0])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels(variants)
    ax.set_ylabel("latency (steps)")
    ax.set_title("p50 and p99 latency distribution by variant")
    return _save(fig, out)

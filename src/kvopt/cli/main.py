"""Typer CLI for kv-cache-optimizer."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kvopt.runner import sweep

app = typer.Typer(no_args_is_help=True, help="KV-cache attention-variant benchmark.")
console = Console()


@app.command()
def bench(
    out_dir: Path = typer.Option(Path("runs/latest")),
    n_requests: int = typer.Option(200),
    seed: int = typer.Option(17),
) -> None:
    """Sweep variants x evictions x capacities; write summary.json + 5 PNGs."""
    result = sweep(out_dir, n_requests=n_requests, seed=seed)
    console.print_json(json.dumps({"n_runs": result["n_runs"]}, default=str))


@app.command()
def report(out_dir: Path = typer.Option(Path("runs/latest"))) -> None:
    """Pretty-print the best (variant, eviction) per metric."""
    data = json.loads((out_dir / "summary.json").read_text())
    table = Table(title="Per-run results")
    for col in (
        "variant",
        "eviction",
        "throughput_tps",
        "peak_kv_mb",
        "p50",
        "p99",
        "evicted",
    ):
        table.add_column(col)
    for r in data["results"]:
        table.add_row(
            r["variant"],
            r["eviction"],
            f"{r['throughput_tps']:.2f}",
            f"{r['peak_kv_mb']:.1f}",
            f"{r['p50_latency_steps']:.1f}",
            f"{r['p99_latency_steps']:.1f}",
            str(r["evicted"]),
        )
    console.print(table)


if __name__ == "__main__":
    app()

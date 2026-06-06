"""Discrete-step KV-cache simulator.

The simulator runs one request at a time per step (a simplification chosen to
keep the model legible; batching is reflected in the attention variant's
prefill/decode multipliers, not in per-step concurrency). At each step:

  1. Admit newly-arrived requests, evicting if necessary.
  2. Issue one decode token for every active session whose prefill has
     completed.
  3. Retire any session whose output length has been reached.

The output captures peak KV-bytes, throughput in tokens-per-step, and the
p50/p99 of per-request completion latency in steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvopt.attention.variants import profile
from kvopt.eviction.policies import Session, policy_for
from kvopt.types import AttentionVariant, EvictionPolicy, Request, RunResult


@dataclass
class CacheConfig:
    capacity_bytes: int


def simulate(
    reqs: list[Request],
    variant: AttentionVariant,
    eviction: EvictionPolicy,
    capacity_mb: int = 64,
    max_steps: int = 5000,
) -> RunResult:
    prof = profile(variant)
    cap = CacheConfig(capacity_bytes=capacity_mb * 1024 * 1024)
    policy = policy_for(eviction)

    by_arrival: dict[int, list[Request]] = {}
    for r in reqs:
        by_arrival.setdefault(r.arrival_step, []).append(r)

    sessions: dict[int, Session] = {}
    prefill_done: dict[int, int] = {}
    output_emitted: dict[int, int] = {}
    completion_step: dict[int, int] = {}
    pending: dict[int, Request] = {r.id: r for r in reqs}
    peak_bytes = 0
    evicted_total = 0
    tokens_emitted = 0

    for step in range(max_steps):
        # 1) admit any new requests, evict if needed
        for r in by_arrival.get(step, []):
            need = int(r.prompt_tokens * prof.bytes_per_token)
            while (
                sum(s.kv_bytes for s in sessions.values()) + need > cap.capacity_bytes and sessions
            ):
                victim_id = policy.select(sessions, step)
                evicted_total += 1
                pending.pop(victim_id, None)
                sessions.pop(victim_id, None)
                prefill_done.pop(victim_id, None)
                output_emitted.pop(victim_id, None)
            if need > cap.capacity_bytes:
                continue
            sessions[r.id] = Session(
                id=r.id,
                kv_bytes=need,
                last_access_step=step,
                access_count=1,
                insertion_step=step,
            )
            prefill_done[r.id] = step + max(1, int(r.prompt_tokens / (1024 * prof.prefill_speedup)))
            output_emitted[r.id] = 0

        # 2) decode
        decoded_this_step = 0
        for sid, sess in list(sessions.items()):
            if step < prefill_done.get(sid, step):
                continue
            req = pending.get(sid)
            if req is None:
                continue
            output_emitted[sid] = output_emitted.get(sid, 0) + 1
            tokens_emitted += 1
            decoded_this_step += 1
            sess.last_access_step = step
            sess.access_count += 1
            sess.kv_bytes += int(prof.bytes_per_token)
            if output_emitted[sid] >= req.output_tokens:
                completion_step[sid] = step
                sessions.pop(sid)
                pending.pop(sid)
                prefill_done.pop(sid, None)
                output_emitted.pop(sid, None)

        peak_bytes = max(peak_bytes, sum(s.kv_bytes for s in sessions.values()))

        if not sessions and not any(s > step for s in by_arrival):
            break

    completion_latencies = [
        completion_step[r.id] - r.arrival_step for r in reqs if r.id in completion_step
    ]
    completed = len(completion_latencies)
    arr = np.array(completion_latencies) if completion_latencies else np.array([0.0])

    return RunResult(
        variant=variant,
        eviction=eviction,
        n_requests=len(reqs),
        completed=completed,
        evicted=evicted_total,
        peak_kv_mb=peak_bytes / (1024 * 1024),
        throughput_tps=tokens_emitted / max(1, step + 1),
        p50_latency_steps=float(np.percentile(arr, 50)),
        p99_latency_steps=float(np.percentile(arr, 99)),
    )

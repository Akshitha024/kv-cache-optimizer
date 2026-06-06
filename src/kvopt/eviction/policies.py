"""Eviction policies.

Each policy implements `select(sessions, current_step)` returning the session
id to evict when the cache is full. The simulator calls this when a new
request needs space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kvopt.types import EvictionPolicy


@dataclass
class Session:
    """Per-session state tracked by the cache."""

    id: int
    kv_bytes: int
    last_access_step: int
    access_count: int
    insertion_step: int


class Policy(Protocol):
    def select(self, sessions: dict[int, Session], current_step: int) -> int: ...


@dataclass
class LRUPolicy:
    def select(self, sessions: dict[int, Session], current_step: int) -> int:
        return min(sessions.values(), key=lambda s: s.last_access_step).id


@dataclass
class LFUPolicy:
    def select(self, sessions: dict[int, Session], current_step: int) -> int:
        return min(sessions.values(), key=lambda s: (s.access_count, s.last_access_step)).id


@dataclass
class TimePolicy:
    def select(self, sessions: dict[int, Session], current_step: int) -> int:
        """Time-based: evict the oldest by insertion step."""
        return min(sessions.values(), key=lambda s: s.insertion_step).id


def policy_for(p: EvictionPolicy) -> Policy:
    if p == EvictionPolicy.LRU:
        return LRUPolicy()
    if p == EvictionPolicy.LFU:
        return LFUPolicy()
    return TimePolicy()

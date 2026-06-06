"""Tests for the eviction policies."""

from __future__ import annotations

from kvopt.eviction.policies import LFUPolicy, LRUPolicy, Session, TimePolicy


def _sessions() -> dict[int, Session]:
    return {
        1: Session(id=1, kv_bytes=10, last_access_step=10, access_count=5, insertion_step=0),
        2: Session(id=2, kv_bytes=10, last_access_step=20, access_count=2, insertion_step=5),
        3: Session(id=3, kv_bytes=10, last_access_step=15, access_count=10, insertion_step=2),
    }


def test_lru_evicts_oldest_access() -> None:
    assert LRUPolicy().select(_sessions(), current_step=30) == 1


def test_lfu_evicts_least_used() -> None:
    assert LFUPolicy().select(_sessions(), current_step=30) == 2


def test_time_evicts_oldest_insertion() -> None:
    assert TimePolicy().select(_sessions(), current_step=30) == 1

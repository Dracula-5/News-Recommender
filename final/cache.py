"""
Minimal thread-safe in-memory TTL cache.

This is a deliberate, documented stand-in for Redis: the interface
(``get`` / ``set`` / ``get_or_set``) is the same shape a Redis-backed cache
would have, so swapping the backend later — the moment this needs to run as
more than one process — means changing this one file, not every call site.

Why cache at all here: `/categories` and `/users` hit SQLite on every call
and barely change between requests; `recommend()` re-derives trending +
category-preference aggregates from `interactions`/`category_stats` that
only move a little between feedback events. A short TTL (default 30s,
`NEUROFEED_CACHE_TTL_SECONDS`) removes that redundant work without ever
serving data that's meaningfully stale for a single-user session.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Entry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, default_ttl: float = 30.0) -> None:
        self.default_ttl = default_ttl
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        with self._lock:
            self._store[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + (ttl if ttl is not None else self.default_ttl),
            )

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [k for k in self._store if k.startswith(prefix)]:
                del self._store[key]

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: float | None = None) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value, ttl)
        return value

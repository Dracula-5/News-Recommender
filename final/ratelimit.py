"""
Fixed-window rate limiter, keyed by an arbitrary string (client IP in
practice). Good enough for a single-process API; a fleet deployment would
back this with Redis (`INCR` + `EXPIRE`) behind the same ``allow()`` call.
"""
from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, tuple[int, float]] = {}  # key -> (count, window_start)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            count, window_start = self._buckets.get(key, (0, now))
            if now - window_start >= self.window_seconds:
                count, window_start = 0, now
            count += 1
            self._buckets[key] = (count, window_start)
            return count <= self.max_requests

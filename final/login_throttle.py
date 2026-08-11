"""
Login brute-force throttle, keyed by account email.

The general per-IP rate limiter (ratelimit.py) caps total request volume,
but at its default (120 req/60s) that's still ~120 password guesses a
minute against a single account from one IP, and does nothing at all
against a patient attacker spacing requests below that ceiling or a
distributed one rotating IPs. This is a second, narrower control: after
`max_failures` failed logins for the *same account* within `window_seconds`,
that account is locked out for `lockout_seconds` regardless of source IP —
closing the "just try passwords slowly/from many IPs" gap the general
limiter can't.

Deliberately account-keyed, not IP-keyed: an attacker who wants to lock a
victim out could abuse that (fail a few logins for someone else's email
from anywhere), which is a real trade-off, not an oversight — a short
lockout (60s default) bounds the damage of that griefing path while still
making credential-stuffing meaningfully slower, which is the trade a
system without CAPTCHA/2FA available can reasonably make.
"""
from __future__ import annotations

import threading
import time


class LoginThrottle:
    def __init__(
        self,
        max_failures: int = 5,
        window_seconds: float = 300.0,
        lockout_seconds: float = 60.0,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        # key -> (failure_count, first_failure_time, locked_until)
        self._state: dict[str, tuple[int, float, float]] = {}
        self._lock = threading.Lock()

    def seconds_until_unlocked(self, key: str) -> float:
        """0 if not currently locked out, otherwise seconds remaining."""
        with self._lock:
            entry = self._state.get(key)
            if entry is None:
                return 0.0
            _count, _first, locked_until = entry
            remaining = locked_until - time.monotonic()
            return max(0.0, remaining)

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            count, first_failure, _locked_until = self._state.get(key, (0, now, 0.0))
            if now - first_failure > self.window_seconds:
                count, first_failure = 0, now  # window expired, start fresh
            count += 1
            locked_until = now + self.lockout_seconds if count >= self.max_failures else 0.0
            self._state[key] = (count, first_failure, locked_until)

    def record_success(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)

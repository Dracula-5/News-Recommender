"""
Tiny in-process metrics registry exposed at ``/metrics`` in Prometheus text
exposition format — no ``prometheus_client`` dependency, just enough to
plug this into a Prometheus/Grafana stack (or just eyeball it with curl)
without pulling in a library for what's fundamentally a few counters and
running sums.

Two metric kinds, which is all an API like this needs:
  * counters   — monotonically increasing (request counts, error counts)
  * histograms — summary stats (count/sum/min/max) per labeled series,
                 good enough for "is p-ish latency reasonable" without the
                 bucket-configuration ceremony of a real histogram.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


def _label_key(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


@dataclass
class _HistogramState:
    count: int = 0
    total: float = 0.0
    min: float = field(default=float("inf"))
    max: float = field(default=float("-inf"))


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], float] = {}
        self._histograms: dict[tuple[str, str], _HistogramState] = {}

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = (name, _label_key(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = (name, _label_key(labels))
        with self._lock:
            state = self._histograms.setdefault(key, _HistogramState())
            state.count += 1
            state.total += value
            state.min = min(state.min, value)
            state.max = max(state.max, value)

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, label_key), value in sorted(self._counters.items()):
                metric = f"{name}{{{label_key}}}" if label_key else name
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{metric} {value}")
            for (name, label_key), state in sorted(self._histograms.items()):
                suffix = f"{{{label_key}}}" if label_key else ""
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{suffix} {state.count}")
                lines.append(f"{name}_sum{suffix} {state.total}")
                if state.count:
                    lines.append(f"{name}_avg{suffix} {state.total / state.count}")
                    lines.append(f"{name}_min{suffix} {state.min}")
                    lines.append(f"{name}_max{suffix} {state.max}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()

"""
Mood as an actual scoring signal.

Before this, `mood` was threaded all the way through the stack — the
frontend sends it on every /recommend call, it's in the request schema,
it's stored in `users.mood` — and then silently dropped: `recommend()`
accepted a `mood` parameter and never referenced it again. This module is
what makes "top-k recommendations for the user's latest mood" actually
true instead of aspirational.

Two pieces, same shape as everything else time-sensitive in this codebase
(graph_memory.py's interest decay, rl_policies.py's bandit forgetting):

  1. A heuristic mood -> category affinity prior (MOOD_CATEGORY_AFFINITY).
     Used directly for cold-start-ish freshness and as an interpretable
     starting point; once enough feedback accumulates, the DDQN's own
     mood_affinity input (see recommender.py's 11-D state) can learn
     finer-grained, per-user associations that diverge from this table.
  2. Time decay: a mood is a *momentary* context signal, not a stable
     preference — "curious" declared 10 minutes ago should dominate,
     declared yesterday it shouldn't still be steering the feed. Half-life
     is deliberately much shorter than the graph's (1 week): moods swing
     within a day, topic interests don't.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# category -> affinity in [0, 1]; unlisted categories default to neutral
# (0.5, see mood_affinity()). Values are a starting prior, not a claim
# about what any individual user actually wants — they're deliberately
# mild (0.6-0.9 range) so they nudge rather than dominate the blend.
MOOD_CATEGORY_AFFINITY: dict[str, dict[str, float]] = {
    "curious":    {"science": 0.9, "technology": 0.85, "world": 0.7, "education": 0.7},
    "focused":    {"technology": 0.85, "business": 0.8, "education": 0.75, "science": 0.65},
    "relaxed":    {"lifestyle": 0.9, "entertainment": 0.85, "sports": 0.6},
    "energetic":  {"sports": 0.9, "entertainment": 0.7, "technology": 0.55},
    "stressed":   {"lifestyle": 0.7, "health": 0.65, "entertainment": 0.6},
    "tired":      {"entertainment": 0.7, "lifestyle": 0.65},
    "busy":       {"business": 0.7, "technology": 0.6, "world": 0.55},
    "happy":      {"entertainment": 0.75, "sports": 0.65, "lifestyle": 0.6},
    "anxious":    {"health": 0.65, "lifestyle": 0.6},
}

# Half-life for mood freshness: 4 hours. Contrast with graph_memory.py's
# _HALF_LIFE of 1 week for topic interest — a mood is volatile on purpose.
MOOD_HALF_LIFE_SECONDS: float = 4 * 3600


def mood_affinity(mood: str, category: str) -> float:
    """Raw (undecayed) heuristic affinity in [0, 1]. 0.5 = neutral/no opinion."""
    if not mood:
        return 0.5
    table = MOOD_CATEGORY_AFFINITY.get(mood.strip().lower())
    if not table:
        return 0.5
    return table.get(category, 0.5)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # SQLite CURRENT_TIMESTAMP format: "YYYY-MM-DD HH:MM:SS", naive UTC.
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def mood_decay_factor(mood_updated_at: str | None, now: datetime | None = None) -> float:
    """
    1.0 = just set, decaying toward 0.0 with MOOD_HALF_LIFE_SECONDS.
    0.0 if there's no timestamp at all (mood was never actually set, as
    opposed to having decayed away — the caller should treat these the
    same, but the distinction matters for testing the decay math itself).
    """
    ts = _parse_timestamp(mood_updated_at)
    if ts is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    elapsed = max(0.0, (now - ts).total_seconds())
    return 0.5 ** (elapsed / MOOD_HALF_LIFE_SECONDS)


def decayed_mood_affinity(mood: str, category: str, mood_updated_at: str | None, now: datetime | None = None) -> float:
    """
    The actual per-category signal recommend()/feedback() use: the raw
    affinity, pulled toward neutral (0.5) by however much the mood has
    gone stale. A "curious" mood set 4 hours ago (one half-life) pulls a
    science article halfway between its full 0.9 affinity and neutral;
    set a day ago, it's essentially neutral again.
    """
    raw = mood_affinity(mood, category)
    decay = mood_decay_factor(mood_updated_at, now)
    return 0.5 + (raw - 0.5) * decay

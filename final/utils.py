from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Sequence


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_0_1(value: float, scale: float = 100.0) -> float:
    value = float(value or 0)
    if value > 1.0:
        return clamp(value / scale)
    return clamp(value)


def compute_interaction_score(
    click_score: float,
    poll_val: float,
    dwell_time: float,
    long_term_history: float = 0.2,
) -> float:
    """score = 0.55·click + 0.10·poll + 0.20·log_dwell + 0.15·long_term_history"""
    click   = clamp(click_score)
    poll    = clamp(poll_val)
    # Normalise dwell to [0,1]: 60 s → 1.0
    log_dwell = clamp(math.log(max(0.0, float(dwell_time or 0)) + 1.0) / math.log(61.0))
    history = clamp(long_term_history)
    return round((0.55 * click) + (0.10 * poll) + (0.20 * log_dwell) + (0.15 * history), 3)


def compute_reward(
    liked: int,
    time_spent: float,
    similarity: float = 0.0,
    scroll_depth: float = 0.0,
) -> float:
    """
    Multi-signal HITL reward:
      ±2.0  explicit like / dislike
      ±1.2  dwell-time bonus/penalty  (> 6 s good, < 2 s bad)
      +0.50 similarity match bonus
      +0.40 deep scroll bonus  (scroll_depth > 0.7 → strong engagement)
      −0.20 shallow scroll on skip  (scroll_depth < 0.15)
    Range approx [−3.4, +4.1]
    """
    reward = 2.0 if int(liked) == 1 else -2.0
    if time_spent > 6:
        reward += 1.2
    elif time_spent < 2:
        reward -= 1.2
    reward += 0.50 * clamp(similarity)
    depth = clamp(float(scroll_depth))
    if depth > 0.7:
        reward += 0.40
    elif depth < 0.15 and not liked:
        reward -= 0.20
    return round(reward, 4)


def category_preference_from_text(categories: Sequence[str], text: str) -> dict[str, float]:
    text_l = (text or "").lower()
    prefs = {}
    for cat in categories:
        if cat.lower() in text_l:
            prefs[cat] = 0.9
    if not prefs:
        for cat in categories:
            prefs[cat] = 1.0 / max(len(categories), 1)
    return prefs


def diversity_limited(items: list[dict], k: int, max_same_category: int = 3) -> list[dict]:
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    for item in items:
        cat = item.get("category", "general")
        if counts[cat] >= max_same_category:
            continue
        selected.append(item)
        counts[cat] += 1
        if len(selected) >= k:
            break
    return selected


def lexical_similarity(query: str, docs: Iterable[str]) -> list[float]:
    q_words = {w for w in (query or "").lower().split() if len(w) > 2}
    scores = []
    for doc in docs:
        d_words = {w for w in (doc or "").lower().split() if len(w) > 2}
        if not q_words or not d_words:
            scores.append(0.0)
        else:
            scores.append(len(q_words & d_words) / max(len(q_words | d_words), 1))
    return scores


def small_randomness(scale: float = 0.03) -> float:
    return random.uniform(0.0, scale)

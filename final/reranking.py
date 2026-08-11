"""
Maximal Marginal Relevance (MMR) re-ranking.

Plain top-K-by-score selection tends to pick near-duplicate articles when
several candidates score well for the same reason (e.g. five near-identical
"markets" stories all riding the same trending boost). MMR trades a
controlled amount of relevance for diversity by picking items one at a
time, each time favoring whichever remaining candidate is *both* relevant
and unlike what's already been picked:

    MMR(d) = lambda * relevance(d) - (1 - lambda) * max_sim(d, selected)

lambda=1.0 reduces to plain top-K; lambda=0.0 ignores relevance entirely
and just maximizes spread. `NEUROFEED_MMR_LAMBDA` (default 0.75) leans
toward relevance while still breaking up same-category clusters — this is
the standard Carbonell & Goldstein (1998) formulation, applied here over
LSA content vectors rather than raw text.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def mmr_select(
    items: list[dict[str, Any]],
    k: int,
    *,
    relevance_key: str = "score",
    id_key: str = "news_id",
    vectors: dict[str, np.ndarray] | None = None,
    fallback_similarity: Callable[[dict, dict], float] | None = None,
    lambda_param: float = 0.75,
) -> list[dict[str, Any]]:
    """
    Greedy MMR selection over `items`, returning up to `k` of them in
    picked order (most-relevance-and-diversity-balanced first).

    `vectors` maps id_key -> content embedding (LSA); pairs missing a
    vector fall back to `fallback_similarity(a, b)` (defaults to a
    same-category indicator) so re-ranking degrades gracefully instead of
    crashing when the NLP engine hasn't fitted an item yet.
    """
    if not items:
        return []
    vectors = vectors or {}

    def sim(a: dict, b: dict) -> float:
        va = vectors.get(a[id_key])
        vb = vectors.get(b[id_key])
        if va is not None and vb is not None:
            return _cosine(va, vb)
        if fallback_similarity is not None:
            return fallback_similarity(a, b)
        return 1.0 if a.get("category") == b.get("category") else 0.0

    pool = list(items)
    scores = [float(it.get(relevance_key, 0.0)) for it in pool]
    max_score = max(scores) or 1e-9
    norm_scores = [s / max_score for s in scores]  # relevance term needs [0,1]-ish scale

    selected: list[dict[str, Any]] = []
    selected_idx: list[int] = []
    remaining = list(range(len(pool)))

    while remaining and len(selected) < k:
        best_idx = None
        best_mmr = float("-inf")
        for i in remaining:
            if not selected_idx:
                penalty = 0.0
            else:
                penalty = max(sim(pool[i], pool[j]) for j in selected_idx)
            mmr = lambda_param * norm_scores[i] - (1 - lambda_param) * penalty
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(pool[best_idx])
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return selected

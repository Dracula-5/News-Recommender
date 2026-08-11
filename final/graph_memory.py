"""
User interest knowledge graph for news recommendation.

Each user's reading history is modelled as a directed graph of category
nodes. Node weights (interest score per category) and edge weights
(co-interest between category pairs) both decay exponentially over time
(half-life = 1 week), so recent interactions dominate and stale interests
fade out on their own rather than sticking around forever.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from database import DB_PATH, db_connect

logger = logging.getLogger(__name__)


# Half-life: 1 week in seconds
_HALF_LIFE: float = 86_400 * 7

# Co-interest edges below this weight are pruned lazily on read — long
# enough decayed that they're noise, not signal.
_EDGE_PRUNE_THRESHOLD = 0.01

# Fixed per-interaction weight for the avg_reward EMA — independent of
# elapsed time between interactions (that's what time-decay-toward-neutral
# at read time is for; conflating the two made steady engagement and idle
# decay fight each other — see database.py's note on the avg_reward column).
_EMA_ALPHA = 0.3


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _decay(elapsed_seconds: float) -> float:
    return 0.5 ** (max(0.0, elapsed_seconds) / _HALF_LIFE)


class UserInterestGraph:
    """
    Per-user category interest graph with time-decayed rewards.

    Nodes — one per category:
        total_reward   : decayed cumulative reward signal (display/activity magnitude)
        avg_reward      : fixed-weight EMA of reward — this drives the score
        count           : raw (undecayed) total interactions, for display
        liked           : total liked interactions
        last_time       : unix timestamp of last update

    Edges — category_a -> category_b, undirected in effect (mirrored both
    ways) since "liked A and B together" has no inherent direction:
        weight          : decayed co-interest strength
        last_time       : unix timestamp last reinforced

    Public API:
        record_interaction(user_id, category, reward, liked, dwell_time)
        get_category_score(user_id, category) -> float [0,1]
        get_category_scores(user_id, categories) -> np.ndarray [0,1]
        get_top_categories(user_id, n) -> List[str]
        get_co_interest_categories(user_id, category, n) -> List[str]
        get_graph_stats(user_id) -> dict
        save_to_db(user_id)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._load_from_db()

    # ──────────────────────────────────────────────────────────
    #  Internal helpers (assume the lock is already held — never call
    #  these from outside a `with self._lock:` block, and never call a
    #  *public* method from inside one, or it'll deadlock on the
    #  non-reentrant lock)
    # ──────────────────────────────────────────────────────────

    def _graph(self, user_id: str) -> nx.DiGraph:
        if user_id not in self._graphs:
            self._graphs[user_id] = nx.DiGraph()
        return self._graphs[user_id]

    def _node_score_locked(self, data: dict, now: float | None = None) -> float:
        """
        Convert node attributes to a [0,1] interest score, applying decay
        live for whatever time has elapsed since `last_time` — not just
        the decay baked in as of the last write. Nodes are only ever
        mutated on a write or a DB load; without this, a category a user
        stops interacting with would keep whatever score it had forever,
        which defeats the entire point of "stale interests fade out."

        Deliberately *not* total_reward/count: dividing a decayed sum by
        an ever-growing raw count sinks the average toward zero over a
        long history even with steady, undiminished engagement (numerator
        decays, denominator never does — and a count that decays at the
        *same* rate as the numerator cancels out in the ratio, which
        removes decay-toward-neutral entirely instead of fixing it; both
        were tried and both broke one of the two behaviors this needs).
        avg_reward is a fixed-weight EMA instead — stable under steady
        engagement regardless of interaction count — with idle decay
        applied here, at read time, based on elapsed time alone.
        """
        now = time.time() if now is None else now
        decay = _decay(now - float(data.get("last_time", now)))
        effective_avg = float(data.get("avg_reward", 0.0)) * decay
        return _sigmoid(effective_avg * 0.5)

    def _top_categories_locked(self, G: nx.DiGraph, n: int) -> List[str]:
        if not G.nodes:
            return []
        now = time.time()
        pairs: List[Tuple[str, float]] = [
            (cat, self._node_score_locked(data, now)) for cat, data in G.nodes(data=True)
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in pairs[:n]]

    def _edge_weight_locked(self, G: nx.DiGraph, a: str, b: str, now: float) -> float:
        if not G.has_edge(a, b):
            return 0.0
        edge = G[a][b]
        return float(edge.get("weight", 0.0)) * _decay(now - float(edge.get("last_time", now)))

    # ──────────────────────────────────────────────────────────
    #  Write
    # ──────────────────────────────────────────────────────────

    def record_interaction(
        self,
        user_id: str,
        category: str,
        reward: float,
        liked: bool,
        dwell_time: float = 0.0,
    ) -> None:
        """Record a reading interaction and update the category node."""
        with self._lock:
            G = self._graph(user_id)
            now = time.time()
            if not G.has_node(category):
                G.add_node(
                    category,
                    total_reward=0.0,
                    avg_reward=0.0,
                    count=0,
                    liked=0,
                    last_time=now,
                )
            n = G.nodes[category]
            decay = _decay(now - n["last_time"])
            n["total_reward"] = n["total_reward"] * decay + float(reward)
            # Fixed-weight EMA — NOT decayed by elapsed time here (that's
            # applied separately, at read time, in _node_score_locked).
            n["avg_reward"] = n["avg_reward"] * (1 - _EMA_ALPHA) + float(reward) * _EMA_ALPHA
            n["count"] += 1
            n["liked"] += int(liked)
            n["last_time"] = now

            # Co-interest edge: strengthen the association between this
            # category and every other category the user has liked before.
            # Weight decays independently per edge (see _edge_weight_locked),
            # so an association from months ago that's never reinforced
            # again fades out on its own — recency comes from the decay,
            # not from artificially restricting which pairs get connected.
            if liked:
                for other_cat, odata in list(G.nodes(data=True)):
                    if other_cat == category or odata.get("liked", 0) <= 0:
                        continue
                    for a, b in ((category, other_cat), (other_cat, category)):
                        if G.has_edge(a, b):
                            prior = self._edge_weight_locked(G, a, b, now)
                            G[a][b]["weight"] = min(1.0, prior + 0.05)
                        else:
                            G.add_edge(a, b, weight=0.05)
                        G[a][b]["last_time"] = now

    # ──────────────────────────────────────────────────────────
    #  Read
    # ──────────────────────────────────────────────────────────

    def get_category_score(self, user_id: str, category: str) -> float:
        with self._lock:
            G = self._graph(user_id)
            if not G.has_node(category):
                return 0.5
            return self._node_score_locked(G.nodes[category])

    def get_category_scores(self, user_id: str, categories: List[str]) -> np.ndarray:
        """Return a [0,1] interest score array aligned with `categories`."""
        with self._lock:
            G = self._graph(user_id)
            now = time.time()
            scores = np.zeros(len(categories))
            for i, cat in enumerate(categories):
                scores[i] = self._node_score_locked(G.nodes[cat], now) if G.has_node(cat) else 0.5
            return scores

    def get_top_categories(self, user_id: str, n: int = 5) -> List[str]:
        """Return top-N categories by interest score."""
        with self._lock:
            return self._top_categories_locked(self._graph(user_id), n)

    def get_co_interest_categories(
        self, user_id: str, category: str, n: int = 3
    ) -> List[str]:
        """Categories that co-occur with *category* in the user's like history."""
        with self._lock:
            G = self._graph(user_id)
            if not G.has_node(category):
                return self._top_categories_locked(G, n)
            now = time.time()
            neighbors = sorted(
                (c for c in G.successors(category)),
                key=lambda c: self._edge_weight_locked(G, category, c, now),
                reverse=True,
            )
            neighbors = [
                c for c in neighbors
                if self._edge_weight_locked(G, category, c, now) > _EDGE_PRUNE_THRESHOLD
            ]
            return neighbors[:n]

    # ──────────────────────────────────────────────────────────
    #  Stats (for the API and dashboard)
    # ──────────────────────────────────────────────────────────

    def get_graph_stats(self, user_id: str) -> dict:
        with self._lock:
            G = self._graph(user_id)
            now = time.time()
            cats = [
                {
                    "category": cat,
                    "score": round(self._node_score_locked(data, now), 3),
                    "count": int(data.get("count", 0)),
                    "liked": int(data.get("liked", 0)),
                    "total_reward": round(float(data.get("total_reward", 0.0)), 3),
                }
                for cat, data in G.nodes(data=True)
            ]
            cats.sort(key=lambda x: x["score"], reverse=True)
            edges = [
                {
                    "a": a, "b": b,
                    "weight": round(self._edge_weight_locked(G, a, b, now), 3),
                }
                for a, b in G.edges()
                if self._edge_weight_locked(G, a, b, now) > _EDGE_PRUNE_THRESHOLD
            ]
            edges.sort(key=lambda x: x["weight"], reverse=True)
            return {
                "user_id": user_id,
                "nodes": len(G.nodes),
                "edges": len(edges),
                "categories": cats,
                "co_interest_edges": edges,
            }

    # ──────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────

    def save_to_db(self, user_id: str) -> None:
        with self._lock:
            G = self._graph(user_id)
            node_rows = [
                (
                    user_id,
                    cat,
                    float(data.get("total_reward", 0.0)),
                    int(data.get("count", 0)),
                    int(data.get("liked", 0)),
                    float(data.get("last_time", time.time())),
                    float(data.get("avg_reward", 0.0)),
                )
                for cat, data in G.nodes(data=True)
            ]
            edge_rows = [
                (
                    user_id, a, b,
                    float(data.get("weight", 0.0)),
                    float(data.get("last_time", time.time())),
                )
                for a, b, data in G.edges(data=True)
            ]
        try:
            with db_connect(self.db_path) as conn:
                conn.execute("DELETE FROM graph_edges WHERE user_id = ?", (user_id,))
                conn.executemany(
                    """
                    INSERT INTO graph_edges
                    (user_id, category, total_reward, count, liked, last_time, avg_reward)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    node_rows,
                )
                conn.execute("DELETE FROM graph_co_interest_edges WHERE user_id = ?", (user_id,))
                if edge_rows:
                    conn.executemany(
                        """
                        INSERT INTO graph_co_interest_edges
                        (user_id, category_a, category_b, weight, last_time)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        edge_rows,
                    )
        except Exception:
            logger.exception("graph_memory: save_to_db failed for user %s", user_id)

    def _load_from_db(self) -> None:
        now = time.time()
        try:
            with db_connect(self.db_path) as conn:
                node_rows = conn.execute(
                    "SELECT user_id, category, total_reward, count, liked, last_time, avg_reward "
                    "FROM graph_edges"
                ).fetchall()
                edge_rows = conn.execute(
                    "SELECT user_id, category_a, category_b, weight, last_time "
                    "FROM graph_co_interest_edges"
                ).fetchall()

            for row in node_rows:
                uid = str(row["user_id"])
                decay = _decay(now - float(row["last_time"]))
                G = self._graph(uid)
                G.add_node(
                    str(row["category"]),
                    total_reward=float(row["total_reward"]) * decay,
                    avg_reward=float(row["avg_reward"] or 0.0) * decay,
                    count=int(row["count"]),
                    liked=int(row["liked"]),
                    # Rebase to *now*: the decay above already accounts for
                    # everything up to this load. Keeping the old timestamp
                    # here would make the next record_interaction() decay
                    # the same elapsed time a second time, silently halving
                    # (or worse) a user's interest score on every restart.
                    last_time=now,
                )

            for row in edge_rows:
                uid = str(row["user_id"])
                decay = _decay(now - float(row["last_time"]))
                weight = float(row["weight"]) * decay
                if weight <= _EDGE_PRUNE_THRESHOLD:
                    continue
                G = self._graph(uid)
                G.add_edge(str(row["category_a"]), str(row["category_b"]), weight=weight, last_time=now)
        except Exception:
            logger.exception("graph_memory: _load_from_db failed")

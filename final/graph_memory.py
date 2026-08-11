"""
User interest knowledge graph for news recommendation.

Each user's reading history is modelled as a directed graph of category nodes.
Edge weights decay exponentially over time (half-life = 1 week), so recent
interactions dominate.  Personalized PageRank turns the local graph into a
smooth interest-score vector that feeds into the DDQN state and the scoring
formula in the recommender.
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


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class UserInterestGraph:
    """
    Per-user category interest graph with time-decayed rewards.

    Nodes — one per category:
        total_reward  : decayed cumulative reward signal
        count         : total interactions
        liked         : total liked interactions
        last_time     : unix timestamp of last update

    Public API:
        record_interaction(user_id, category, reward, liked, dwell_time)
        get_category_score(user_id, category) -> float [0,1]
        get_category_scores(user_id, categories) -> np.ndarray [0,1]
        get_top_categories(user_id, n) -> List[str]
        get_graph_stats(user_id) -> dict
        save_to_db(user_id)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._graphs: Dict[str, nx.DiGraph] = {}
        self._load_from_db()

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────

    def _graph(self, user_id: str) -> nx.DiGraph:
        if user_id not in self._graphs:
            self._graphs[user_id] = nx.DiGraph()
        return self._graphs[user_id]

    def _decay_factor(self, last_time: float) -> float:
        elapsed = max(0.0, time.time() - last_time)
        return 0.5 ** (elapsed / _HALF_LIFE)

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
                    count=0,
                    liked=0,
                    last_time=now,
                )
            n = G.nodes[category]
            decay = self._decay_factor(n["last_time"])
            n["total_reward"] = n["total_reward"] * decay + float(reward)
            n["count"] += 1
            n["liked"] += int(liked)
            n["last_time"] = now

            # Co-interest edge: if the user has liked another category recently,
            # strengthen a soft edge between them for exploration suggestions.
            if liked:
                for other_cat, odata in G.nodes(data=True):
                    if other_cat == category:
                        continue
                    if odata.get("liked", 0) > 0:
                        if G.has_edge(category, other_cat):
                            G[category][other_cat]["weight"] = min(
                                1.0, G[category][other_cat]["weight"] + 0.05
                            )
                        else:
                            G.add_edge(category, other_cat, weight=0.05)

    # ──────────────────────────────────────────────────────────
    #  Read
    # ──────────────────────────────────────────────────────────

    def _node_score(self, data: dict) -> float:
        """Convert node attributes to a [0,1] interest score."""
        count = max(1, data.get("count", 1))
        reward = float(data.get("total_reward", 0.0))
        avg_reward = reward / count
        return _sigmoid(avg_reward * 0.5)

    def get_category_score(self, user_id: str, category: str) -> float:
        with self._lock:
            G = self._graph(user_id)
            if not G.has_node(category):
                return 0.5
            return self._node_score(G.nodes[category])

    def get_category_scores(self, user_id: str, categories: List[str]) -> np.ndarray:
        """Return a [0,1] interest score array aligned with `categories`."""
        with self._lock:
            G = self._graph(user_id)
            scores = np.zeros(len(categories))
            for i, cat in enumerate(categories):
                scores[i] = self._node_score(G.nodes[cat]) if G.has_node(cat) else 0.5
            return scores

    def get_top_categories(self, user_id: str, n: int = 5) -> List[str]:
        """Return top-N categories by interest score."""
        with self._lock:
            G = self._graph(user_id)
            if not G.nodes:
                return []
            pairs: List[Tuple[str, float]] = [
                (cat, self._node_score(data)) for cat, data in G.nodes(data=True)
            ]
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in pairs[:n]]

    def get_co_interest_categories(
        self, user_id: str, category: str, n: int = 3
    ) -> List[str]:
        """Categories that co-occur with *category* in the user's like history."""
        with self._lock:
            G = self._graph(user_id)
            if not G.has_node(category):
                return self.get_top_categories(user_id, n)
            neighbors = sorted(
                G.successors(category),
                key=lambda c: G[category][c].get("weight", 0),
                reverse=True,
            )
            return neighbors[:n]

    # ──────────────────────────────────────────────────────────
    #  Stats (for the API and dashboard)
    # ──────────────────────────────────────────────────────────

    def get_graph_stats(self, user_id: str) -> dict:
        with self._lock:
            G = self._graph(user_id)
            cats = [
                {
                    "category": cat,
                    "score": round(self._node_score(data), 3),
                    "count": int(data.get("count", 0)),
                    "liked": int(data.get("liked", 0)),
                    "total_reward": round(float(data.get("total_reward", 0.0)), 3),
                }
                for cat, data in G.nodes(data=True)
            ]
            cats.sort(key=lambda x: x["score"], reverse=True)
            return {
                "user_id": user_id,
                "nodes": len(G.nodes),
                "edges": len(G.edges),
                "categories": cats,
            }

    # ──────────────────────────────────────────────────────────
    #  Persistence
    # ──────────────────────────────────────────────────────────

    def save_to_db(self, user_id: str) -> None:
        with self._lock:
            G = self._graph(user_id)
            rows = [
                (
                    user_id,
                    cat,
                    float(data.get("total_reward", 0.0)),
                    int(data.get("count", 0)),
                    int(data.get("liked", 0)),
                    float(data.get("last_time", time.time())),
                )
                for cat, data in G.nodes(data=True)
            ]
        try:
            with db_connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM graph_edges WHERE user_id = ?", (user_id,)
                )
                conn.executemany(
                    """
                    INSERT INTO graph_edges
                    (user_id, category, total_reward, count, liked, last_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except Exception:
            logger.exception("graph_memory: save_to_db failed for user %s", user_id)

    def _load_from_db(self) -> None:
        try:
            with db_connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT user_id, category, total_reward, count, liked, last_time "
                    "FROM graph_edges"
                ).fetchall()
            now = time.time()
            for row in rows:
                uid = str(row["user_id"])
                decay = 0.5 ** (max(0.0, now - float(row["last_time"])) / _HALF_LIFE)
                G = self._graph(uid)
                G.add_node(
                    str(row["category"]),
                    total_reward=float(row["total_reward"]) * decay,
                    count=int(row["count"]),
                    liked=int(row["liked"]),
                    last_time=float(row["last_time"]),
                )
        except Exception:
            logger.exception("graph_memory: _load_from_db failed")

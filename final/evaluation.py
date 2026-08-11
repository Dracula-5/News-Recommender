"""
Offline evaluation harness for the recommender.

The tricky part of evaluating *this* system offline is that `recommend()`
deliberately never re-shows an already-seen article — so replaying real
logged interactions and checking "did we recommend what they later liked"
doesn't work; those items are excluded from candidates by design.

Instead this runs a **simulated-user episode**, the standard offline
evaluation pattern for interactive/bandit-style recommenders (see e.g. the
"replay"/simulation methodology used to evaluate contextual bandits):
give a synthetic user a ground-truth set of preferred categories, drive the
real recommend -> feedback loop against the real engine and database for N
rounds, and score each round against that ground truth. Two things fall
out of this for free:

  * standard ranking metrics (Precision@K, NDCG@K) that don't depend on
    circular "did we show them what we showed them" logic
  * a learning curve — precision in the first few rounds vs. the last few
    — which is the actual point: it demonstrates the DDQN/bandit/graph
    stack adapts within a session instead of just statically filtering.

Usage:
    python evaluation.py                 # quick run, prints a report
    python evaluation.py --rounds 20 --users 5
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Any

from recommender import NewsRecommender


# ── Metrics ─────────────────────────────────────────────────────────

def precision_at_k(items: list[dict], relevant_categories: set[str]) -> float:
    if not items:
        return 0.0
    hits = sum(1 for it in items if it["category"] in relevant_categories)
    return hits / len(items)


def ndcg_at_k(items: list[dict], relevant_categories: set[str]) -> float:
    """Binary relevance (1 if category matches the user's true interests)."""
    if not items:
        return 0.0
    dcg = sum(
        (1.0 if it["category"] in relevant_categories else 0.0) / math.log2(rank + 2)
        for rank, it in enumerate(items)
    )
    ideal_hits = min(len(items), sum(1 for it in items if it["category"] in relevant_categories))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits)) or 1e-9
    return dcg / idcg


def category_diversity(items: list[dict]) -> float:
    """Distinct categories / batch size — 1.0 means every item is a different category."""
    if not items:
        return 0.0
    return len({it["category"] for it in items}) / len(items)


def catalog_coverage(seen_news_ids: set[str], catalog_size: int) -> float:
    if catalog_size == 0:
        return 0.0
    return len(seen_news_ids) / catalog_size


# ── Simulated user ──────────────────────────────────────────────────

@dataclass
class SimulatedUser:
    user_id: str
    true_interests: set[str]
    like_prob_in_interest: float = 0.85
    like_prob_outside: float = 0.10


@dataclass
class EpisodeResult:
    user_id: str
    per_round_precision: list[float] = field(default_factory=list)
    per_round_ndcg: list[float] = field(default_factory=list)
    per_round_diversity: list[float] = field(default_factory=list)
    catalog_coverage: float = 0.0

    @property
    def early_precision(self) -> float:
        n = max(1, len(self.per_round_precision) // 3)
        return sum(self.per_round_precision[:n]) / n

    @property
    def late_precision(self) -> float:
        n = max(1, len(self.per_round_precision) // 3)
        return sum(self.per_round_precision[-n:]) / n


def run_episode(
    engine: NewsRecommender,
    sim_user: SimulatedUser,
    rounds: int = 10,
    k: int = 8,
    rng: random.Random | None = None,
) -> EpisodeResult:
    rng = rng or random.Random(0)
    engine.onboard_user(
        {
            "user_id": sim_user.user_id,
            "interests": list(sim_user.true_interests),
            "exploration_preference": 0.25,
        }
    )

    result = EpisodeResult(user_id=sim_user.user_id)
    seen_ids: set[str] = set()

    for _ in range(rounds):
        response = engine.recommend(sim_user.user_id, k=k)
        items = response["recommendations"]
        if not items:
            break

        result.per_round_precision.append(precision_at_k(items, sim_user.true_interests))
        result.per_round_ndcg.append(ndcg_at_k(items, sim_user.true_interests))
        result.per_round_diversity.append(category_diversity(items))

        for item in items:
            seen_ids.add(item["news_id"])
            in_interest = item["category"] in sim_user.true_interests
            like_prob = sim_user.like_prob_in_interest if in_interest else sim_user.like_prob_outside
            liked = 1 if rng.random() < like_prob else 0
            engine.feedback(
                {
                    "user_id": sim_user.user_id,
                    "news_id": item["news_id"],
                    "time_spent": rng.uniform(3, 25) if liked else rng.uniform(0.5, 4),
                    "liked": liked,
                    "skipped": 0 if liked else 1,
                    "scroll_depth": rng.uniform(0.6, 1.0) if liked else rng.uniform(0.0, 0.3),
                    "click_val": liked,
                    "attention_score": rng.uniform(0.5, 0.9) if liked else rng.uniform(0.1, 0.4),
                    "final_score": item.get("score", 0),
                    "similarity": item.get("similarity", 0),
                    "trending": item.get("trending", 0),
                }
            )

    catalog_size = len(engine.categories()) and _catalog_size(engine)
    result.catalog_coverage = catalog_coverage(seen_ids, catalog_size)
    return result


def _catalog_size(engine: NewsRecommender) -> int:
    from database import db_connect

    with db_connect(engine.db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]


# ── Multi-user aggregate run ────────────────────────────────────────

def evaluate(
    engine: NewsRecommender,
    n_users: int = 3,
    rounds: int = 10,
    k: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    categories = engine.categories()
    if not categories:
        raise RuntimeError("No categories in DB — seed demo data first (scripts/seed_demo_data.py).")

    episodes: list[EpisodeResult] = []
    for i in range(n_users):
        true_interests = set(rng.sample(categories, min(2, len(categories))))
        sim_user = SimulatedUser(user_id=f"eval_sim_user_{i}_{seed}", true_interests=true_interests)
        episodes.append(run_episode(engine, sim_user, rounds=rounds, k=k, rng=rng))

    def _avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_users": n_users,
        "rounds_per_user": rounds,
        "k": k,
        "precision_at_k": _avg([p for ep in episodes for p in ep.per_round_precision]),
        "ndcg_at_k": _avg([p for ep in episodes for p in ep.per_round_ndcg]),
        "category_diversity": _avg([p for ep in episodes for p in ep.per_round_diversity]),
        "catalog_coverage": _avg([ep.catalog_coverage for ep in episodes]),
        "early_precision_at_k": _avg([ep.early_precision for ep in episodes]),
        "late_precision_at_k": _avg([ep.late_precision for ep in episodes]),
        "learning_delta": round(
            _avg([ep.late_precision for ep in episodes]) - _avg([ep.early_precision for ep in episodes]), 4
        ),
        "per_user": [
            {
                "user_id": ep.user_id,
                "early_precision_at_k": round(ep.early_precision, 4),
                "late_precision_at_k": round(ep.late_precision, 4),
                "catalog_coverage": round(ep.catalog_coverage, 4),
            }
            for ep in episodes
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    engine = NewsRecommender()
    report = evaluate(engine, n_users=args.users, rounds=args.rounds, k=args.k, seed=args.seed)

    print(f"\nEvaluation over {report['n_users']} simulated users x {report['rounds_per_user']} rounds (k={report['k']})\n")
    print(f"  Precision@K          : {report['precision_at_k']:.3f}")
    print(f"  NDCG@K               : {report['ndcg_at_k']:.3f}")
    print(f"  Category diversity   : {report['category_diversity']:.3f}  (1.0 = every item a different category)")
    print(f"  Catalog coverage     : {report['catalog_coverage']:.3f}")
    print(f"  Precision@K (early)  : {report['early_precision_at_k']:.3f}")
    print(f"  Precision@K (late)   : {report['late_precision_at_k']:.3f}")
    print(f"  Learning delta       : {report['learning_delta']:+.3f}  (late - early; positive = adapting within-session)")


if __name__ == "__main__":
    main()

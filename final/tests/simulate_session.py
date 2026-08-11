from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import DB_PATH, import_csvs
from recommender import NewsRecommender


def run() -> None:
    import_csvs(DB_PATH, force=False)
    engine = NewsRecommender(DB_PATH)
    user_id = "simulation_user"
    engine.onboard_user(
        {
            "user_id": user_id,
            "interests": ["health", "technology"],
            "mood": "focused",
            "time_available": 15,
            "time_of_day": "morning",
            "location": "India",
            "exploration_preference": 0.2,
            "sample_click": "health",
        }
    )

    first = engine.recommend(user_id, k=8)["recommendations"]
    print("Initial categories:", [r["category"] for r in first])

    for _ in range(24):
        rec = engine.recommend(user_id, k=8)["recommendations"][0]
        like = int(rec["category"] in {"health", "technology"})
        engine.feedback(
            {
                "user_id": user_id,
                "news_id": rec["news_id"],
                "time_spent": random.uniform(8, 15) if like else random.uniform(0.5, 1.5),
                "liked": like,
                "skipped": 0 if like else 1,
                "scroll_depth": 0.85 if like else 0.1,
                "click_val": like,
                "attention_score": 0.75 if like else 0.25,
                "final_score": rec["score"],
                "similarity": rec["similarity"],
                "trending": rec["trending"],
            }
        )

    final = engine.recommend(user_id, k=8)["recommendations"]
    counts = {}
    for item in final:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    assert len(final) >= 8
    assert max(counts.values()) <= 3
    assert len({r["news_id"] for r in final}) == len(final)
    print("Final categories:", [r["category"] for r in final])
    print("Checks passed: no repetition, diversity cap, model updates completed.")


if __name__ == "__main__":
    run()

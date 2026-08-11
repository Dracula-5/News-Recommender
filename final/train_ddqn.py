from __future__ import annotations

import argparse
import random

from database import DB_PATH, import_csvs
from recommender import NewsRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed SQLite and warm-start the DDQN with simulated HITL feedback.")
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--force-import", action="store_true")
    args = parser.parse_args()

    import_csvs(DB_PATH, force=args.force_import)
    engine = NewsRecommender(DB_PATH)
    engine.onboard_user(
        {
            "user_id": args.user_id,
            "interests": ["technology", "health", "sports"],
            "mood": "curious",
            "time_available": 12,
            "time_of_day": "evening",
            "location": "India",
            "exploration_preference": 0.25,
            "sample_click": "technology",
        }
    )

    for i in range(args.episodes):
        recs = engine.recommend(args.user_id, k=8)["recommendations"]
        if not recs:
            break
        item = recs[0]
        liked = int(item["category"] in {"technology", "health", "sports"} or random.random() < 0.25)
        time_spent = random.uniform(7, 18) if liked else random.uniform(0.5, 3.0)
        result = engine.feedback(
            {
                "user_id": args.user_id,
                "news_id": item["news_id"],
                "time_spent": time_spent,
                "liked": liked,
                "skipped": 0 if liked else 1,
                "scroll_depth": random.uniform(0.55, 1.0) if liked else random.uniform(0.0, 0.35),
                "click_val": liked,
                "attention_score": random.uniform(0.55, 0.9) if liked else random.uniform(0.2, 0.5),
                "final_score": item["score"],
                "similarity": item["similarity"],
                "trending": item["trending"],
            }
        )
        print(
            f"{i + 1:03d} category={result['category']:<15} liked={liked} "
            f"reward={result['reward']:<5} loss={result['loss']}"
        )


if __name__ == "__main__":
    main()

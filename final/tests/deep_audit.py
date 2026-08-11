from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database import DB_PATH, db_connect
from recommender import NewsRecommender


def assert_no_more_than_three(items: list[dict]) -> None:
    counts = collections.Counter(item["category"] for item in items)
    assert max(counts.values()) <= 3, counts


def audit_dynamic_session() -> None:
    engine = NewsRecommender()
    user_id = f"deep_audit_{int(time.time())}"
    engine.onboard_user(
        {
            "user_id": user_id,
            "interests": ["health", "sports", "finance"],
            "mood": "focused",
            "time_available": 15,
            "time_of_day": "morning",
            "location": "India",
            "exploration_preference": 0.25,
            "sample_click": "sports",
        }
    )

    with db_connect(DB_PATH) as conn:
        before = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_id = ?", (user_id,)).fetchone()[0]
        prefs_before = {
            row["category"]: row["preference"]
            for row in conn.execute(
                "SELECT category, preference FROM user_category_prefs WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        }

    seen_news: list[str] = []
    sources = collections.Counter()
    losses = []
    latencies = []

    for _ in range(20):
        start = time.perf_counter()
        rec = engine.recommend(user_id, 8)
        latencies.append(time.perf_counter() - start)
        items = rec["recommendations"]
        assert len(items) == 8
        assert_no_more_than_three(items)
        sources.update(item["source_mix"] for item in items)

        item = items[0]
        assert item["news_id"] not in seen_news
        seen_news.append(item["news_id"])
        liked = int(item["category"] in {"health", "sports", "finance"})
        result = engine.feedback(
            {
                "user_id": user_id,
                "news_id": item["news_id"],
                "time_spent": 9 if liked else 1,
                "liked": liked,
                "skipped": 0 if liked else 1,
                "scroll_depth": 0.85 if liked else 0.1,
                "click_val": liked,
                "attention_score": 0.8 if liked else 0.25,
                "normalized_attention": 0.8 if liked else 0.25,
                "brightness": 75 if liked else 35,
                "eye_openness": 88 if liked else 45,
                "movement": 12 if liked else 70,
                "energy_score": 82 if liked else 32,
                "gaze_score": 0.78 if liked else 0.25,
                "face_detected": 1,
                "attention_source": "audit",
                "final_score": item["score"],
                "similarity": item["similarity"],
                "trending": item["trending"],
            }
        )
        losses.append(result["loss"])

    with db_connect(DB_PATH) as conn:
        after = conn.execute("SELECT COUNT(*) FROM interactions WHERE user_id = ?", (user_id,)).fetchone()[0]
        memory = conn.execute("SELECT COUNT(*) FROM user_memory WHERE user_id = ?", (user_id,)).fetchone()[0]
        attention_rows = conn.execute("SELECT COUNT(*) FROM attention_events WHERE user_id = ?", (user_id,)).fetchone()[0]
        last_attention = dict(
            conn.execute(
                """
                SELECT brightness, eye_openness, movement, energy_score, gaze_score,
                       normalized_attention, source
                FROM attention_events
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        )
        prefs_after = {
            row["category"]: row["preference"]
            for row in conn.execute(
                "SELECT category, preference FROM user_category_prefs WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        }

    assert after - before == 20
    assert memory == 20
    assert attention_rows == 20
    assert last_attention["source"] == "audit"
    assert 0 <= last_attention["normalized_attention"] <= 1
    assert len(set(seen_news)) == 20
    assert {"personalized", "trending", "random"}.issubset(set(sources))
    assert losses[14] is None
    assert losses[15] is not None
    assert max(latencies) < 2.0, latencies
    assert any(prefs_after.get(cat, 0) >= prefs_before.get(cat, 0) for cat in ["health", "sports", "finance"])

    print("Dynamic session audit passed")
    print("sources:", dict(sources))
    print("max_latency:", round(max(latencies), 3))
    print("loss_at_16:", losses[15])


def audit_blocking() -> None:
    engine = NewsRecommender()
    user_id = f"block_audit_{int(time.time())}"
    engine.onboard_user({"user_id": user_id, "interests": ["health"], "sample_click": "health"})
    blocked_category = None

    for _ in range(2):
        items = engine.recommend(user_id, 8)["recommendations"]
        item = next((x for x in items if blocked_category is None or x["category"] == blocked_category), items[0])
        blocked_category = item["category"]
        engine.feedback(
            {
                "user_id": user_id,
                "news_id": item["news_id"],
                "time_spent": 1,
                "liked": 0,
                "skipped": 1,
                "scroll_depth": 0,
                "click_val": 0,
                "attention_score": 0.2,
                "final_score": item["score"],
                "similarity": item["similarity"],
                "trending": item["trending"],
            }
        )

    rec = engine.recommend(user_id, 8)
    assert blocked_category in rec["blocked_categories"]
    assert blocked_category not in [item["category"] for item in rec["recommendations"]]
    print("Blocking audit passed:", blocked_category)


if __name__ == "__main__":
    audit_dynamic_session()
    audit_blocking()

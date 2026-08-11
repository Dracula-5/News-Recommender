"""
Integration tests against the real engine + a seeded, isolated SQLite DB
(see conftest.py). These supersede the old standalone scripts
(deep_audit.py / simulate_session.py) — same checks, now runnable via
`pytest` with proper isolation instead of mutating the dev database.
"""
from __future__ import annotations

import collections
import time

import pytest

from database import db_connect


def test_recommend_returns_items_within_category_cap(engine, unique_user_id):
    engine.onboard_user({
        "user_id": unique_user_id,
        "interests": ["health", "sports", "technology"],
        "exploration_preference": 0.25,
    })
    items = engine.recommend(unique_user_id, k=8)["recommendations"]
    assert len(items) >= 6
    counts = collections.Counter(item["category"] for item in items)
    assert max(counts.values()) <= 2  # cat_count cap enforced in recommend()
    assert len({item["news_id"] for item in items}) == len(items)  # no duplicates


def test_recommend_never_repeats_an_article_already_given_feedback(engine, unique_user_id):
    # user_memory (the seen-exclusion list) is only populated on feedback,
    # not on mere display — so the contract is "won't re-show anything
    # you've reacted to," not "won't re-show anything ever rendered."
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})
    given_feedback: set[str] = set()
    for _ in range(5):
        items = engine.recommend(unique_user_id, k=8)["recommendations"]
        ids = {i["news_id"] for i in items}
        assert not (ids & given_feedback), "recommend() re-showed an article already given feedback"
        item = items[0]
        given_feedback.add(item["news_id"])
        engine.feedback({
            "user_id": unique_user_id, "news_id": item["news_id"],
            "time_spent": 5, "liked": 1, "final_score": item["score"],
            "similarity": item["similarity"], "trending": item["trending"],
        })


def test_feedback_updates_interaction_count_and_memory(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["health"]})
    with db_connect(db_path) as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE user_id = ?", (unique_user_id,)
        ).fetchone()[0]

    for _ in range(5):
        item = engine.recommend(unique_user_id, k=8)["recommendations"][0]
        engine.feedback({
            "user_id": unique_user_id, "news_id": item["news_id"],
            "time_spent": 8, "liked": 1, "final_score": item["score"],
            "similarity": item["similarity"], "trending": item["trending"],
        })

    with db_connect(db_path) as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE user_id = ?", (unique_user_id,)
        ).fetchone()[0]
        memory = conn.execute(
            "SELECT COUNT(*) FROM user_memory WHERE user_id = ?", (unique_user_id,)
        ).fetchone()[0]
    assert after - before == 5
    assert memory == 5


def test_two_dislikes_temporarily_block_a_category(engine, unique_user_id, db_path):
    # Drive this directly against two known articles from the same
    # category, rather than hoping recommend() keeps surfacing that
    # category after the first dislike already suppresses its score —
    # it's the category-blocking mechanism under test, not the ranker.
    engine.onboard_user({"user_id": unique_user_id, "interests": ["health"]})
    with db_connect(db_path) as conn:
        rows = conn.execute(
            "SELECT news_id FROM news WHERE category = 'health' LIMIT 2"
        ).fetchall()
    news_ids = [r["news_id"] for r in rows]
    assert len(news_ids) == 2

    for news_id in news_ids:
        engine.feedback({
            "user_id": unique_user_id, "news_id": news_id,
            "time_spent": 1, "liked": 0, "skipped": 1,
            "final_score": 0, "similarity": 0, "trending": 0,
        })

    rec = engine.recommend(unique_user_id, 8)
    assert "health" in rec["blocked_categories"]
    assert "health" not in [item["category"] for item in rec["recommendations"]]


def test_cold_start_flag_flips_after_threshold(engine, unique_user_id):
    from config import get_settings

    threshold = get_settings().cold_start_interaction_threshold
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})

    first = engine.recommend(unique_user_id, k=8)
    assert first["is_cold_start"] is True
    assert first["interaction_count"] == 0

    for _ in range(threshold + 1):
        item = engine.recommend(unique_user_id, k=8)["recommendations"][0]
        engine.feedback({
            "user_id": unique_user_id, "news_id": item["news_id"],
            "time_spent": 6, "liked": 1, "final_score": item["score"],
            "similarity": item["similarity"], "trending": item["trending"],
        })

    last = engine.recommend(unique_user_id, k=8)
    assert last["is_cold_start"] is False
    assert last["interaction_count"] >= threshold


def test_a_category_with_mostly_likes_is_not_also_classified_disliked(engine, unique_user_id, db_path):
    """
    Regression test for the bug the evaluation harness caught: a category
    that's liked most of the time in the recent window must not also land
    in disliked_cats just because one interaction among many was a miss
    (binary "any liked / any disliked" let both +0.20 and -0.50 apply to
    the same category — net punishing a user's actual favorite).
    """
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})

    # 4 likes, 1 dislike, all in "technology" -> ratio 0.8, should be
    # classified liked-only, never disliked.
    with db_connect(db_path) as conn:
        for i, liked in enumerate([1, 1, 1, 1, 0]):
            conn.execute(
                "INSERT INTO interactions (user_id, news_id, category, liked) VALUES (?, ?, ?, ?)",
                (unique_user_id, f"synthetic_{i}", "technology", liked),
            )

    recent_rows = [{"category": "technology", "liked": v} for v in [1, 1, 1, 1, 0]]
    cat_like_ratio: dict[str, list[int]] = {}
    for r in recent_rows:
        counts = cat_like_ratio.setdefault(r["category"], [0, 0])
        counts[1] += 1
        counts[0] += int(r["liked"] == 1)
    liked_cats = {c for c, (likes, total) in cat_like_ratio.items() if likes / total >= 0.6}
    disliked_cats = {c for c, (likes, total) in cat_like_ratio.items() if likes / total <= 0.3}

    assert "technology" in liked_cats
    assert "technology" not in disliked_cats

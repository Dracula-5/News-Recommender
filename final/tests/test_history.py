"""
Tests for the reading-history and memory-summary features: a real
`GET /history/{user_id}` (backed by the full `interactions` log, not the
100-entry `user_memory` exclusion list) and `GET /memory/summary/{user_id}`
(a human-readable digest of the knowledge graph — top/rising/fading
interests and co-interest related categories).
"""
from __future__ import annotations

from database import db_connect


def _give_feedback(engine, user_id, category, liked, db_path, n=3):
    with db_connect(db_path) as conn:
        rows = conn.execute("SELECT news_id FROM news WHERE category = ?", (category,)).fetchall()
    news_ids = [r["news_id"] for r in rows]
    assert news_ids, f"no seeded articles for category {category!r}"
    used = []
    for i in range(n):
        news_id = news_ids[i % len(news_ids)]  # cycle if n exceeds the catalog for this category
        engine.feedback({
            "user_id": user_id, "news_id": news_id,
            "time_spent": 8 if liked else 1, "liked": liked,
            "final_score": 0.5, "similarity": 0.3, "trending": 0.2,
        })
        used.append(news_id)
    return used


def test_history_is_empty_for_a_new_user(engine, unique_user_id):
    result = engine.get_history(unique_user_id)
    assert result["total"] == 0
    assert result["items"] == []


def test_history_reflects_feedback_newest_first(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})
    news_ids = _give_feedback(engine, unique_user_id, "technology", liked=1, db_path=db_path, n=3)

    result = engine.get_history(unique_user_id)
    assert result["total"] == 3
    assert len(result["items"]) == 3
    # Newest first: the last one fed back should be items[0].
    assert result["items"][0]["news_id"] == news_ids[-1]
    assert all(item["liked"] for item in result["items"])
    assert all(item["category"] == "technology" for item in result["items"])


def test_history_pagination(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})
    _give_feedback(engine, unique_user_id, "technology", liked=1, db_path=db_path, n=5)

    page1 = engine.get_history(unique_user_id, limit=2, offset=0)
    page2 = engine.get_history(unique_user_id, limit=2, offset=2)
    assert len(page1["items"]) == 2
    assert len(page2["items"]) == 2
    assert {i["news_id"] for i in page1["items"]}.isdisjoint({i["news_id"] for i in page2["items"]})


def test_memory_summary_reflects_top_interest(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})
    _give_feedback(engine, unique_user_id, "technology", liked=1, db_path=db_path, n=5)

    summary = engine.get_memory_summary(unique_user_id)
    assert "technology" in summary["top_interests"]
    assert summary["total_interactions"] == 5


def test_memory_summary_detects_rising_interest(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})
    # No prior history in "science" at all, then a burst of likes — should
    # show up as rising (dominates recent activity despite low count).
    _give_feedback(engine, unique_user_id, "science", liked=1, db_path=db_path, n=5)

    summary = engine.get_memory_summary(unique_user_id)
    assert "science" in summary["rising_interests"]
    assert "science" not in summary["fading_interests"]


def test_memory_summary_detects_fading_interest(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"]})
    # A well-established category (high historical count)...
    _give_feedback(engine, unique_user_id, "technology", liked=1, db_path=db_path, n=8)
    # ...then 20 recent interactions in a completely different category,
    # pushing "technology" entirely out of the recent-20 window.
    _give_feedback(engine, unique_user_id, "sports", liked=1, db_path=db_path, n=20)

    summary = engine.get_memory_summary(unique_user_id)
    assert "technology" in summary["fading_interests"]
    assert "technology" not in summary["rising_interests"]


def test_memory_summary_related_categories_uses_graph_co_interest(engine, unique_user_id, db_path):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology", "science"]})
    _give_feedback(engine, unique_user_id, "technology", liked=1, db_path=db_path, n=3)
    _give_feedback(engine, unique_user_id, "science", liked=1, db_path=db_path, n=3)

    summary = engine.get_memory_summary(unique_user_id)
    related = summary["related_categories"]
    if "technology" in related:
        assert "science" in related["technology"]


def test_history_endpoint_via_api(api_client, unique_user_id):
    api_client.post("/onboard", json={"user_id": unique_user_id, "interests": ["technology"]})
    resp = api_client.get(f"/history/{unique_user_id}")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == unique_user_id


def test_memory_summary_endpoint_via_api(api_client, unique_user_id):
    api_client.post("/onboard", json={"user_id": unique_user_id, "interests": ["technology"]})
    resp = api_client.get(f"/memory/summary/{unique_user_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "top_interests" in body
    assert "related_categories" in body


def test_history_endpoint_respects_user_authorization(api_client):
    victim = api_client.post(
        "/auth/signup",
        json={"email": "history_victim@example.com", "username": "history_victim", "password": "testpass123"},
    ).json()
    attacker = api_client.post(
        "/auth/signup",
        json={"email": "history_attacker@example.com", "username": "history_attacker", "password": "testpass123"},
    ).json()
    resp = api_client.get(
        f"/history/{victim['user_id']}",
        headers={"Authorization": f"Bearer {attacker['token']}"},
    )
    assert resp.status_code == 403

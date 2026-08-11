"""
FastAPI TestClient smoke tests — one full request/response cycle through
each route family, against the isolated test DB from conftest.py.
"""
from __future__ import annotations


def test_health_reports_ok_with_seeded_data(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db"]["news"] > 0
    assert body["db"]["users"] > 0


def test_categories_endpoint(api_client):
    resp = api_client.get("/categories")
    assert resp.status_code == 200
    assert len(resp.json()["categories"]) > 0


def test_metrics_endpoint_is_prometheus_text(api_client):
    api_client.get("/categories")  # generate at least one data point
    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text


def test_onboard_recommend_feedback_roundtrip(api_client, unique_user_id):
    onboard = api_client.post(
        "/onboard",
        json={"user_id": unique_user_id, "interests": ["technology", "sports"], "exploration_preference": 0.3},
    )
    assert onboard.status_code == 200

    rec = api_client.post("/recommend", json={"user_id": unique_user_id, "k": 8})
    assert rec.status_code == 200
    items = rec.json()["recommendations"]
    assert len(items) > 0
    assert rec.json()["is_cold_start"] is True

    item = items[0]
    fb = api_client.post(
        "/feedback",
        json={
            "user_id": unique_user_id,
            "news_id": item["news_id"],
            "time_spent": 6,
            "liked": 1,
            "final_score": item["score"],
            "similarity": item["similarity"],
            "trending": item["trending"],
        },
    )
    assert fb.status_code == 200


def test_recommend_for_unknown_user_auto_onboards(api_client, unique_user_id):
    resp = api_client.post("/recommend", json={"user_id": unique_user_id, "k": 8})
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) > 0


def test_eval_run_endpoint_returns_metrics(api_client):
    resp = api_client.get("/eval/run", params={"users": 1, "rounds": 2, "k": 8})
    assert resp.status_code == 200
    body = resp.json()
    for key in ("precision_at_k", "ndcg_at_k", "category_diversity", "catalog_coverage"):
        assert key in body
        assert 0.0 <= body[key] <= 1.0 + 1e-9


def test_trending_endpoint_reflects_feedback(api_client, unique_user_id):
    # Generate at least one interaction so category_stats.trending_score > 0
    # for something — a brand-new seeded DB starts with everything at 0.
    api_client.post("/onboard", json={"user_id": unique_user_id, "interests": ["technology"]})
    rec = api_client.post("/recommend", json={"user_id": unique_user_id, "k": 8}).json()
    item = rec["recommendations"][0]
    api_client.post("/feedback", json={
        "user_id": unique_user_id, "news_id": item["news_id"],
        "time_spent": 10, "liked": 1,
        "final_score": item["score"], "similarity": item["similarity"], "trending": item["trending"],
    })

    resp = api_client.get("/trending")
    assert resp.status_code == 200
    sections = resp.json()["trending"]
    assert len(sections) > 0
    for section in sections:
        assert section["trending_score"] > 0
        assert len(section["articles"]) > 0
        assert all(a["category"] == section["category"] for a in section["articles"])


def test_trending_endpoint_respects_limits(api_client):
    resp = api_client.get("/trending", params={"limit_categories": 2, "articles_per_category": 1})
    assert resp.status_code == 200
    sections = resp.json()["trending"]
    assert len(sections) <= 2
    assert all(len(s["articles"]) <= 1 for s in sections)

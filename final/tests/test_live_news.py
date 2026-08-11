"""
Tests for the Live News bulletin (live_news.py + GET /news/live).

live_news.py talks to a real external API (NewsAPI.org), which the test
suite must never actually call — every test here monkeypatches
`live_news.requests.get` instead. Covers: the no-key no-op path (the
default/expected state until NEUROFEED_NEWS_API_KEY is set), category
classification, upsert-by-URL dedup, and the exclusion of is_live=1 rows
from recommend()'s personalized candidate pool.
"""
from __future__ import annotations

from config import Settings
from database import db_connect
from live_news import _classify_category, fetch_and_store_live_news, fetch_live_articles


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _settings_with_key(**overrides):
    return Settings(news_api_key="fake-test-key", **overrides)


def _article(url="https://example.com/story-1", title="A technology breakthrough", **overrides):
    base = {
        "title": title,
        "description": "Researchers announce a new technology breakthrough.",
        "content": "Full content of the technology breakthrough story goes here.",
        "url": url,
        "urlToImage": "https://example.com/img.jpg",
        "publishedAt": "2026-01-01T00:00:00Z",
        "source": {"name": "Example Wire"},
    }
    base.update(overrides)
    return base


# ── classification ──────────────────────────────────────────

def test_classify_category_matches_keyword_in_title():
    cats = ["technology", "sports", "world"]
    assert _classify_category("Big technology announcement", "", cats) == "technology"


def test_classify_category_matches_keyword_in_description():
    cats = ["technology", "sports", "world"]
    assert _classify_category("Big game tonight", "sports fans excited", cats) == "sports"


def test_classify_category_falls_back_to_world_when_no_match():
    cats = ["technology", "sports", "world"]
    assert _classify_category("Nothing relevant here", "", cats) == "world"


# ── fetch_live_articles ─────────────────────────────────────

def test_fetch_live_articles_is_a_noop_without_a_key(monkeypatch):
    called = False

    def fake_get(*a, **k):
        nonlocal called
        called = True
        return _FakeResponse({"status": "ok", "articles": [_article()]})

    monkeypatch.setattr("live_news.requests.get", fake_get)
    settings = Settings(news_api_key="")
    assert fetch_live_articles(settings) == []
    assert not called, "should never call the external API with no key configured"


def test_fetch_live_articles_returns_empty_on_request_failure(monkeypatch):
    def fake_get(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr("live_news.requests.get", fake_get)
    assert fetch_live_articles(_settings_with_key()) == []


def test_fetch_live_articles_returns_empty_on_api_error_status(monkeypatch):
    monkeypatch.setattr(
        "live_news.requests.get",
        lambda *a, **k: _FakeResponse({"status": "error", "message": "bad key"}),
    )
    assert fetch_live_articles(_settings_with_key()) == []


# ── fetch_and_store_live_news (DB integration) ──────────────

def test_stores_live_articles_tagged_is_live(monkeypatch, db_path):
    monkeypatch.setattr(
        "live_news.requests.get",
        lambda *a, **k: _FakeResponse({"status": "ok", "articles": [_article()]}),
    )
    stored = fetch_and_store_live_news(db_path, _settings_with_key())
    assert len(stored) == 1
    news_id = stored[0]["news_id"]

    with db_connect(db_path) as conn:
        row = conn.execute("SELECT * FROM news WHERE news_id = ?", (news_id,)).fetchone()
    assert row is not None
    assert row["is_live"] == 1
    assert row["source_name"] == "Example Wire"
    assert row["url"] == "https://example.com/story-1"
    assert row["category"] == "technology"

    # Cleanup — this test writes into the shared session-scoped test DB.
    with db_connect(db_path) as conn:
        conn.execute("DELETE FROM news WHERE news_id = ?", (news_id,))


def test_refetching_the_same_url_upserts_not_duplicates(monkeypatch, db_path):
    art = _article(title="Original headline")
    monkeypatch.setattr(
        "live_news.requests.get",
        lambda *a, **k: _FakeResponse({"status": "ok", "articles": [art]}),
    )
    first = fetch_and_store_live_news(db_path, _settings_with_key())
    news_id = first[0]["news_id"]

    updated = _article(title="Original headline", description="An updated description.")
    monkeypatch.setattr(
        "live_news.requests.get",
        lambda *a, **k: _FakeResponse({"status": "ok", "articles": [updated]}),
    )
    second = fetch_and_store_live_news(db_path, _settings_with_key())
    assert second[0]["news_id"] == news_id  # same URL -> same stable id

    with db_connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM news WHERE news_id = ?", (news_id,)).fetchone()[0]
        row = conn.execute("SELECT abstract FROM news WHERE news_id = ?", (news_id,)).fetchone()
    assert count == 1, "re-fetching the same URL must upsert, not duplicate"
    assert "updated description" in row["abstract"]

    with db_connect(db_path) as conn:
        conn.execute("DELETE FROM news WHERE news_id = ?", (news_id,))


def test_skips_articles_missing_title_or_url(monkeypatch, db_path):
    removed = _article(url="https://example.com/removed", title="[Removed]")
    removed["title"] = ""  # NewsAPI's "[Removed]" placeholders are inconsistent about which field is blanked
    monkeypatch.setattr(
        "live_news.requests.get",
        lambda *a, **k: _FakeResponse({"status": "ok", "articles": [removed]}),
    )
    stored = fetch_and_store_live_news(db_path, _settings_with_key())
    assert stored == []


# ── exclusion from the personalized pipeline ────────────────

def test_live_articles_never_appear_in_recommend_candidates(monkeypatch, engine, unique_user_id, db_path):
    monkeypatch.setattr(
        "live_news.requests.get",
        lambda *a, **k: _FakeResponse({"status": "ok", "articles": [_article(url="https://example.com/story-2")]}),
    )
    stored = fetch_and_store_live_news(db_path, _settings_with_key())
    live_news_id = stored[0]["news_id"]

    engine.onboard_user({"user_id": unique_user_id, "interests": ["technology"], "exploration_preference": 0.3})
    seen_ids = set()
    for _ in range(5):
        items = engine.recommend(unique_user_id, k=8)["recommendations"]
        seen_ids.update(i["news_id"] for i in items)

    assert live_news_id not in seen_ids

    with db_connect(db_path) as conn:
        conn.execute("DELETE FROM news WHERE news_id = ?", (live_news_id,))


# ── GET /news/live ───────────────────────────────────────────

def test_news_live_endpoint_reports_unconfigured_without_a_key(api_client):
    # conftest.py never sets NEUROFEED_NEWS_API_KEY, so this reflects the
    # real default state until a key is added on Render.
    resp = api_client.get("/news/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["configured"] is False
    assert body["articles"] == []

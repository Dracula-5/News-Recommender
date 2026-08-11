"""
Live news ingestion (NewsAPI.org) — the "Live News" bulletin/postcard row.

One call to NewsAPI's top-headlines endpoint per refresh (gated by a TTL
cache in backend/main.py, default 90 min — see config.py's
`news_live_cache_ttl_seconds`), classified into this app's category
vocabulary with the same simple substring heuristic `utils.py` already uses
for onboarding preferences, then upserted into the existing `news` table
tagged `is_live=1`.

Deliberately display-only: `recommend()`'s candidate queries and
`get_trending()`'s sample picker both exclude `is_live=1` rows (see
database.py's migration comment) — these are surfaced exclusively through
`GET /news/live`, not folded into the personalized/trained pipeline.

No API key configured -> every function here is a documented no-op, never
a startup error. Network/API failures degrade the same way (log a warning,
return/keep whatever was already persisted) — the bulletin just goes quiet
rather than breaking the rest of the app.
"""
from __future__ import annotations

import hashlib
import logging

import requests

from config import Settings
from database import db_connect

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 6.0


def _news_id_for(url: str) -> str:
    # Stable hash of the article URL so re-fetching the same story upserts
    # in place instead of accumulating duplicates across refreshes.
    return "live_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _classify_category(title: str, description: str, categories: list[str]) -> str:
    text = f"{title or ''} {description or ''}".lower()
    for cat in categories:
        if cat.lower() in text:
            return cat
    return "world" if "world" in categories else (categories[0] if categories else "world")


def fetch_live_articles(settings: Settings) -> list[dict]:
    """Raw NewsAPI.org call. Returns [] if unconfigured or the request fails."""
    if not settings.news_api_key:
        return []
    try:
        resp = requests.get(
            settings.news_api_base_url,
            params={
                "apiKey": settings.news_api_key,
                "country": settings.news_api_country,
                "pageSize": settings.news_api_page_size,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network error, timeout, non-2xx, bad JSON — all non-fatal here
        logger.warning("live_news: fetch failed: %s", e)
        return []

    if data.get("status") != "ok":
        logger.warning("live_news: API returned status=%s: %s", data.get("status"), data.get("message"))
        return []
    return data.get("articles", []) or []


def fetch_and_store_live_news(db_path, settings: Settings) -> list[dict]:
    """
    Fetches, classifies, and upserts live articles into `news`. Returns the
    list of stored rows (empty if unconfigured, the fetch failed, or the
    API returned nothing usable this cycle — the DB simply keeps whatever
    was persisted on the last successful refresh).
    """
    articles = fetch_live_articles(settings)
    if not articles:
        return []

    from recommender import NewsRecommender  # local import: avoid a module-level cycle

    categories = NewsRecommender(db_path).categories()
    stored: list[dict] = []

    with db_connect(db_path) as conn:
        for a in articles:
            url = (a.get("url") or "").strip()
            title = (a.get("title") or "").strip()
            if not url or not title:
                continue  # NewsAPI occasionally returns "[Removed]" placeholder entries

            description = (a.get("description") or "").strip()
            content = (a.get("content") or "").strip()
            full_article = content or description or title
            source_name = ((a.get("source") or {}).get("name") or "").strip()
            category = _classify_category(title, description, categories)
            news_id = _news_id_for(url)

            row = {
                "news_id":      news_id,
                "category":     category,
                "abstract":     description or title,
                "full_article": full_article,
                "url":          url,
                "image_url":    a.get("urlToImage") or "",
                "source_name":  source_name,
                "published_at": a.get("publishedAt") or "",
                "article_length": len(full_article.split()),
            }
            conn.execute(
                """
                INSERT INTO news
                (news_id, category, subcategory, location, preferred_time, age_group,
                 article_length, abstract, full_article, url, is_live, image_url,
                 source_name, published_at)
                VALUES (?, ?, '', '', '', '', ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(news_id) DO UPDATE SET
                    category=excluded.category, abstract=excluded.abstract,
                    full_article=excluded.full_article, url=excluded.url,
                    article_length=excluded.article_length, image_url=excluded.image_url,
                    source_name=excluded.source_name, published_at=excluded.published_at
                """,
                (
                    row["news_id"], row["category"], row["article_length"],
                    row["abstract"], row["full_article"], row["url"],
                    row["image_url"], row["source_name"], row["published_at"],
                ),
            )
            stored.append(row)

    logger.info("live_news: refreshed %d articles", len(stored))
    return stored

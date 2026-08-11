"""
Centralized runtime configuration.

Every tunable that used to be a hardcoded constant or a scattered
`os.getenv(...)` call lives here instead, validated once at startup via
pydantic-settings. Override any of it with environment variables (see
`.env.example`) — nothing here requires touching code to change behavior
between local dev, CI, and a container deployment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEUROFEED_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment ─────────────────────────────────────────
    env: str = "development"          # development | production | test
    log_level: str = "INFO"

    # ── Data ────────────────────────────────────────────────
    db_path: Path = ROOT / "data" / "news_recommender.sqlite"
    auth_db_path: Path = ROOT / "data" / "auth.sqlite"

    # ── CORS ────────────────────────────────────────────────
    # Comma-separated origins in production; "*" is fine for local dev only.
    cors_origins: str = "*"

    # ── Webcam attention ────────────────────────────────────
    enable_webcam_attention: bool = False

    # ── Recommender tuning ──────────────────────────────────
    # Below this many logged interactions a user is treated as cold-start:
    # scoring leans harder on stated interests + trending, lighter on the
    # (still mostly untrained) RL/bandit signals.
    cold_start_interaction_threshold: int = 5

    # MMR (Maximal Marginal Relevance) diversity re-ranking trade-off.
    # 1.0 = pure relevance ranking, 0.0 = pure novelty/diversity.
    mmr_lambda: float = 0.75

    # Max distinct users' DDQNAgent (PyTorch model + bandit state) kept
    # resident in memory at once. Without a bound this grows for every
    # distinct user_id a long-running process ever serves — least-recently
    # used agents are checkpointed to disk and evicted past this cap, and
    # transparently reloaded from disk the next time that user is active.
    max_cached_agents: int = 500

    # ── Caching ─────────────────────────────────────────────
    cache_ttl_seconds: float = 30.0

    # ── API hardening ───────────────────────────────────────
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0

    # Stricter limiter applied only to /auth/signup and /auth/login — these
    # are higher-value abuse targets (credential stuffing, fake account
    # creation) than the general API surface, so they get a tighter budget
    # on top of (not instead of) the general per-IP limiter above.
    auth_rate_limit_requests: int = 10
    auth_rate_limit_window_seconds: float = 60.0

    # Login brute-force lockout — see login_throttle.py.
    login_max_failures: int = 5
    login_failure_window_seconds: float = 300.0
    login_lockout_seconds: float = 60.0

    # ── Live news (NewsAPI.org) ─────────────────────────────
    # Empty by default — the "Live News" bulletin (GET /news/live) and
    # the background refresh it triggers are both no-ops without a key,
    # never a startup error. Get a free key at https://newsapi.org.
    news_api_key: str = ""
    news_api_base_url: str = "https://newsapi.org/v2/top-headlines"
    news_api_country: str = "us"
    # How many articles to pull per refresh. Kept small deliberately —
    # this is a display bulletin, not the main catalog.
    news_api_page_size: int = 12
    # Minimum time between live refreshes. NewsAPI's free tier caps out at
    # 100 requests/day; one refresh = one request, so 90 min (16/day) stays
    # well under that with room for manual restarts.
    news_live_cache_ttl_seconds: float = 5400.0

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

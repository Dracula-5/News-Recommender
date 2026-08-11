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

    # ── Caching ─────────────────────────────────────────────
    cache_ttl_seconds: float = 30.0

    # ── API hardening ───────────────────────────────────────
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

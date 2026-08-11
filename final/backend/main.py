from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cache import TTLCache
from config import get_settings
from logging_config import configure_logging, get_logger
from metrics import METRICS
from ratelimit import RateLimiter
from database import db_connect, import_csvs, init_db
from recommender import NewsRecommender
from utils import category_preference_from_text, clamp
from webcam_attention import (
    get_latest_attention_snapshot,
    get_latest_frame,
    start_background_attention,
    stop_background_attention,
)
from auth import init_auth_db, signup_user, login_user, verify_token, get_user_by_id

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

DB_PATH = settings.db_path

# ─────────────────────────────────────────────
#  WARMUP (Performance Boost)
# ─────────────────────────────────────────────

async def _warmup():
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _warmup_sync)
    except Exception:
        logger.exception("Warmup failed")


def _warmup_sync():
    engine._ensure_nlp()
    with db_connect(DB_PATH) as conn:
        conn.execute("SELECT news_id, category, subcategory, abstract FROM news").fetchall()
        conn.execute("SELECT * FROM users LIMIT 500").fetchall()
        conn.execute("SELECT category, trending_score FROM category_stats").fetchall()
        first_user = conn.execute("SELECT user_id FROM users LIMIT 1").fetchone()
    if first_user:
        engine._agent(first_user["user_id"])


# ─────────────────────────────────────────────
#  APP LIFECYCLE
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting backend (env=%s, db=%s)", settings.env, DB_PATH)

    init_db(DB_PATH)
    init_auth_db()

    if settings.enable_webcam_attention:
        try:
            result = start_background_attention()
            logger.info("Webcam attention started: %s", result)
        except Exception:
            logger.exception("Webcam attention failed to start")
    else:
        logger.info("Webcam attention disabled (NEUROFEED_ENABLE_WEBCAM_ATTENTION=false)")

    asyncio.ensure_future(_warmup())
    yield

    logger.info("Shutting down")
    stop_background_attention()


# ─────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="HITL DDQN News Recommender",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory TTL cache for hot, cheap-to-stale read paths (categories, users
# list). Swap for Redis behind the same get/set interface if this ever runs
# as more than one process — see cache.py.
cache = TTLCache(default_ttl=settings.cache_ttl_seconds)

# Simple in-memory sliding-window rate limiter, keyed by client IP. Good
# enough for a single-process deployment; a multi-instance deployment would
# back this with Redis (the interface is intentionally the same shape).
rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Rate limiting + request logging + latency metrics in one pass."""
    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.allow(client_ip):
        METRICS.inc("http_requests_rate_limited_total")
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS.inc("http_requests_errors_total")
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        raise
    duration_ms = (time.perf_counter() - start) * 1000

    METRICS.inc("http_requests_total")
    METRICS.observe("http_request_duration_ms", duration_ms, labels={"path": request.url.path})
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response

# ─────────────────────────────────────────────
#  FRONTEND SERVE
# ─────────────────────────────────────────────

frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")

engine = NewsRecommender(DB_PATH)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def as_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────

class OnboardPayload(BaseModel):
    user_id: str | None = None
    interests: list[str] = Field(default_factory=list)
    mood: str = ""
    time_available: float = 10
    time_of_day: str = ""
    location: str = ""
    exploration_preference: float = 0.25
    sample_click: str = ""


class RecommendPayload(BaseModel):
    user_id: str
    k: int = 8
    mode: str | None = None
    location: str | None = None
    mood: str | None = None


class FeedbackPayload(BaseModel):
    user_id: str
    news_id: str
    time_spent: float = 0
    liked: int = 0
    skipped: int = 0
    scroll_depth: float = 0
    click_val: float = 0
    attention_score: float = 0.5
    normalized_attention: float | None = None
    brightness: float = 50
    eye_openness: float = 70
    movement: float = 20
    energy_score: float = 63.5
    gaze_score: float = 0.58
    face_detected: int = 0
    attention_source: str = "frontend"
    final_score: float = 0
    similarity: float = 0
    trending: float = 0
    poll_val: float = 0.5


class PollFeedbackPayload(BaseModel):
    user_id: str
    news_id: str
    liked: int


class UpdatePayload(BaseModel):
    user_id: str


class InterestsUpdatePayload(BaseModel):
    interests: list[str] = Field(default_factory=list)
    mood: str = ""
    exploration_preference: float | None = None


# ─────────────────────────────────────────────
#  AUTH SCHEMAS
# ─────────────────────────────────────────────

class SignupPayload(BaseModel):
    email: str
    username: str
    password: str


class LoginPayload(BaseModel):
    email: str
    password: str


class NlpScorePayload(BaseModel):
    user_id: str
    news_ids: list[str]


# ─────────────────────────────────────────────
#  BASIC ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Open /app for UI or /docs for API."}


@app.get("/health")
def health():
    """Detailed system health — use to verify all components are ready."""
    import time as _time
    snap = get_latest_attention_snapshot()
    try:
        with db_connect(DB_PATH) as conn:
            news_count   = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
            user_count   = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            inter_count  = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            graph_count  = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        db_ok = True
    except Exception as exc:
        news_count = user_count = inter_count = graph_count = 0
        db_ok = False

    return {
        "status":   "ok" if db_ok else "degraded",
        "ts":       _time.time(),
        "db": {
            "ok":           db_ok,
            "news":         news_count,
            "users":        user_count,
            "interactions": inter_count,
            "graph_edges":  graph_count,
        },
        "nlp": {
            "fitted":    engine._nlp_fitted,
            "bm25_docs": len(engine._nlp.bm25.doc_ids),
            "n_lsa":     engine._nlp.n_lsa,
        },
        "rl": {
            "agents_loaded": len(engine._agents),
            "categories":    len(engine.categories()),
        },
        "webcam": {
            "active":         snap.get("status") == "running",
            "face_detected":  bool(snap.get("face_detected")),
            "attention":      snap.get("normalized_attention", 0),
        },
    }


@app.get("/metrics")
def metrics():
    """Prometheus-format request counters/latency — scrape this or curl it directly."""
    return PlainTextResponse(METRICS.render_prometheus(), media_type="text/plain; version=0.0.4")


@app.post("/import_data")
def import_data(force: bool = False):
    return import_csvs(DB_PATH, force=force)


@app.get("/categories")
def categories():
    # Rarely changes within a run — cheap to cache for the full TTL.
    return {"categories": cache.get_or_set("categories", engine.categories)}


@app.get("/users")
def users():
    def _fetch():
        with db_connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT user_id FROM users ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
        return [row["user_id"] for row in rows]

    return {"users": cache.get_or_set("users:recent", _fetch)}


@app.get("/user/{user_id}")
def user_info(user_id: str):
    with db_connect(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)


@app.patch("/user/{user_id}/interests")
def update_interests(user_id: str, payload: InterestsUpdatePayload):
    """
    Update a user's interests, mood, and/or exploration preference after onboarding.
    Re-seeds category preferences from the new interest text.
    """
    with db_connect(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "User not found")

        categories    = engine.categories()
        interest_text = " ".join(payload.interests + [payload.mood])
        prefs = category_preference_from_text(categories, interest_text) if payload.interests else {}

        updates: list[str] = []
        params: list = []
        if payload.interests:
            updates.append("user_interest_text = ?")
            params.append(interest_text)
        if payload.mood:
            updates.append("mood = ?")
            params.append(payload.mood)
        if payload.exploration_preference is not None:
            updates.append("exploration_signal = ?")
            params.append(clamp(payload.exploration_preference))
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(user_id)
            conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", params
            )

        if payload.interests and prefs:
            for cat, pref in prefs.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO user_category_prefs
                    (user_id, category, preference, dislikes, blocked_until_step, updated_at)
                    VALUES (?, ?, ?, 0, 0, CURRENT_TIMESTAMP)
                    """,
                    (user_id, cat, pref),
                )

    return {
        "user_id":      user_id,
        "interests":    payload.interests,
        "mood":         payload.mood,
        "message":      "interests updated",
        "categories_seeded": len(prefs) if payload.interests else 0,
    }


# ─────────────────────────────────────────────
#  AUTHENTICATION ROUTES
# ─────────────────────────────────────────────

@app.post("/auth/signup")
def signup(payload: SignupPayload):
    """
    Create a new user account
    
    Returns:
        {
            'user_id': str,
            'username': str,
            'email': str,
            'token': str
        }
    """
    email = payload.email.strip().lower()
    username = payload.username.strip()
    password = payload.password
    
    # Validation
    if not email or '@' not in email:
        raise HTTPException(400, "Invalid email format")
    
    if not username or len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    
    if not re.match(r'^[A-Za-z0-9_]+$', username):
        raise HTTPException(400, "Username may only contain letters, numbers, and underscores")
    
    if not password or len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    
    result = signup_user(email, password, username)
    
    if not result['success']:
        raise HTTPException(400, result['message'])
    
    # Auto-onboard the user in the recommendation system
    engine.onboard_user({
        'user_id': result['user_id'],
        'interests': [],
        'mood': '',
        'time_available': 10,
        'exploration_preference': 0.25
    })
    cache.invalidate("users:recent")

    return {
        'user_id': result['user_id'],
        'username': result['username'],
        'email': result['email'],
        'token': result['token']
    }


@app.post("/auth/login")
def login(payload: LoginPayload):
    """
    Authenticate user and return token
    
    Returns:
        {
            'user_id': str,
            'username': str,
            'email': str,
            'token': str
        }
    """
    email = payload.email.strip().lower()
    password = payload.password
    
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    
    result = login_user(email, password)
    
    if not result['success']:
        raise HTTPException(401, result['message'])
    
    return {
        'user_id': result['user_id'],
        'username': result['username'],
        'email': result['email'],
        'token': result['token']
    }


@app.get("/auth/user/{user_id}")
def get_auth_user(user_id: str):
    """Get authenticated user info"""
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user


@app.post("/auth/verify")
def verify_auth_token(token: str = None):
    """Verify authentication token"""
    if not token:
        raise HTTPException(400, "Token required")
    
    result = verify_token(token)
    
    if not result['valid']:
        raise HTTPException(401, result.get('message', 'Invalid token'))
    
    return {'valid': True, 'user_id': result['user_id']}


# ─────────────────────────────────────────────
#  CORE RL ROUTES
# ─────────────────────────────────────────────

@app.post("/onboard")
def onboard(payload: OnboardPayload):
    result = engine.onboard_user(as_payload(payload))
    cache.invalidate("users:recent")
    return result


@app.post("/recommend")
def recommend(payload: RecommendPayload):
    return engine.recommend(
        payload.user_id,
        payload.k,
        mode=payload.mode,
        location=payload.location,
        mood=payload.mood,
    )


@app.post("/feedback")
def feedback(payload: FeedbackPayload):
    return engine.feedback(as_payload(payload))


@app.post("/poll_feedback")
def poll_feedback(payload: PollFeedbackPayload):
    return engine.poll_feedback(payload.user_id, payload.news_id, payload.liked)


@app.post("/update_model")
def update_model(payload: UpdatePayload):
    return engine.update_model(payload.user_id)


# ─────────────────────────────────────────────
#  ATTENTION (FIXED)
# ─────────────────────────────────────────────

@app.get("/attention")
def attention():
    return get_latest_attention_snapshot()


@app.get("/attention/frame")
def attention_frame():
    frame = get_latest_frame()
    if not frame:
        return Response(status_code=204)
    return Response(content=frame, media_type="image/jpeg")


@app.get("/attention/stream")
async def attention_stream():
    async def generate():
        while True:
            frame = get_latest_frame()
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────
#  CONTROL ROUTES
# ─────────────────────────────────────────────

@app.post("/attention/start")
def start_attention():
    return start_background_attention()


@app.post("/attention/stop")
def stop_attention():
    return stop_background_attention()


# ─────────────────────────────────────────────
#  RL METRICS
# ─────────────────────────────────────────────

@app.get("/rl/metrics/{user_id}")
def rl_metrics(user_id: str):
    """
    Full RL dashboard payload for a user:
      - DDQN step, epsilon, replay buffer size
      - HybridPolicy blend weight (DDQN vs Thompson Sampling)
      - Thompson Sampling expected reward + 95% CI per category
      - UCB visit counts and values per category
    """
    agent = engine._agent(user_id)
    cats  = agent.categories
    cis   = agent.hybrid.get_ts_confidence(cats)
    return {
        "user_id":        user_id,
        "step":           agent.step,
        "epsilon":        round(agent.epsilon, 4),
        "replay_size":    len(agent.replay),
        "ddqn_weight":    round(agent.hybrid.ddqn_weight(), 3),
        "hybrid_step":    agent.hybrid.step,
        "categories":     cats,
        "ts_confidence":  cis,
        "ucb_values":     [round(float(v), 4) for v in agent.hybrid.ucb.values],
        "ucb_counts":     [int(c) for c in agent.hybrid.ucb.counts],
    }


# ─────────────────────────────────────────────
#  GRAPH MEMORY
# ─────────────────────────────────────────────

@app.get("/graph/stats/{user_id}")
def graph_stats(user_id: str):
    """
    User's knowledge-graph interest scores per category.
    Score [0,1] — higher = stronger interest inferred from graph.
    Includes temporal-decay so stale interactions fade out.
    """
    return engine._graph.get_graph_stats(user_id)


@app.get("/graph/top/{user_id}")
def graph_top_categories(user_id: str, n: int = 5):
    """Top-N categories by graph interest score."""
    top  = engine._graph.get_top_categories(user_id, n=n)
    cats = engine.categories()
    out  = [
        {
            "category": cat,
            "score":    round(engine._graph.get_category_score(user_id, cat), 4),
        }
        for cat in top
        if cat in cats
    ]
    return {"user_id": user_id, "top_categories": out}


# ─────────────────────────────────────────────
#  NLP / SIMILARITY DEBUG
# ─────────────────────────────────────────────

@app.get("/nlp/status")
def nlp_status():
    """Check whether the enhanced NLP engine has been fitted."""
    return {
        "fitted":    engine._nlp_fitted,
        "n_lsa":     engine._nlp.n_lsa,
        "bm25_docs": len(engine._nlp.bm25.doc_ids),
    }


@app.post("/nlp/score")
def nlp_score(payload: NlpScorePayload):
    """
    Score a list of article IDs against the user's interest text.
    Useful for debugging retrieval quality.
    """
    with db_connect(DB_PATH) as conn:
        user = conn.execute(
            "SELECT user_interest_text FROM users WHERE user_id = ?", (payload.user_id,)
        ).fetchone()
    if not user:
        raise HTTPException(404, "User not found")
    engine._ensure_nlp()
    scores = engine._nlp.score_candidates(user["user_interest_text"], payload.news_ids)
    return {"user_id": payload.user_id, "scores": scores}
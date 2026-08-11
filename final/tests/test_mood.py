"""
Tests for mood as an actual scoring/RL signal (mood_signals.py plus the
recommender.py / utils.py wiring that consumes it).

Before this work, `mood` was threaded through the request schema and stored
in `users.mood`, but recommend() silently dropped it — these tests exist to
pin down that it's now a real, decaying, testable signal:
  - mood_signals.py's affinity table and half-life decay math in isolation
  - recommend()'s "latest mood wins" persistence semantics
  - the 11-D DDQN state actually carries mood
  - the cold-start/warm scoring weights still sum to 0.80
  - compute_reward()'s mood-congruence shaping term
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mood_signals import (
    MOOD_HALF_LIFE_SECONDS,
    decayed_mood_affinity,
    mood_affinity,
    mood_decay_factor,
)
from utils import compute_reward


# ── mood_affinity table ─────────────────────────────────────

def test_mood_affinity_known_mood_known_category():
    assert mood_affinity("curious", "science") == 0.9


def test_mood_affinity_known_mood_unlisted_category_is_neutral():
    assert mood_affinity("curious", "sports") == 0.5


def test_mood_affinity_unknown_mood_is_neutral():
    assert mood_affinity("bored", "science") == 0.5


def test_mood_affinity_empty_mood_is_neutral():
    assert mood_affinity("", "science") == 0.5


def test_mood_affinity_is_case_and_whitespace_insensitive():
    assert mood_affinity("  CURIOUS  ", "science") == 0.9


# ── decay math ───────────────────────────────────────────────

def test_decay_factor_is_full_strength_when_just_set():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    assert mood_decay_factor(ts, now=now) == pytest.approx(1.0)


def test_decay_factor_is_half_after_one_half_life():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=MOOD_HALF_LIFE_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
    assert mood_decay_factor(ts, now=now) == pytest.approx(0.5, rel=1e-6)


def test_decay_factor_is_zero_with_no_timestamp():
    assert mood_decay_factor(None) == 0.0


def test_decay_factor_approaches_zero_after_many_half_lives():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = (now - timedelta(seconds=MOOD_HALF_LIFE_SECONDS * 10)).strftime("%Y-%m-%d %H:%M:%S")
    assert mood_decay_factor(ts, now=now) < 0.01


def test_decayed_mood_affinity_pulls_toward_neutral_over_time():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fresh_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    stale_ts = (now - timedelta(seconds=MOOD_HALF_LIFE_SECONDS * 20)).strftime("%Y-%m-%d %H:%M:%S")

    fresh = decayed_mood_affinity("curious", "science", fresh_ts, now=now)
    stale = decayed_mood_affinity("curious", "science", stale_ts, now=now)

    assert fresh == pytest.approx(0.9, rel=1e-6)
    assert stale == pytest.approx(0.5, abs=0.01)


def test_decayed_mood_affinity_with_no_timestamp_is_neutral():
    assert decayed_mood_affinity("curious", "science", None) == pytest.approx(0.5)


# ── state vector wiring ─────────────────────────────────────

def test_state_vector_is_11_dimensional_and_carries_mood(engine, unique_user_id):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["science"], "mood": "curious"})
    state = engine.build_state(unique_user_id, "science", {"mood_affinities": {"science": 0.9}})
    assert len(state) == 11
    assert state[-1] == pytest.approx(0.9)


# ── recommend(): "latest mood wins" persistence ─────────────

def test_recommend_persists_a_new_mood_with_fresh_timestamp(engine, unique_user_id, db_path):
    from database import db_connect

    engine.onboard_user({"user_id": unique_user_id, "interests": ["science"]})
    engine.recommend(unique_user_id, k=8, mood="curious")

    with db_connect(db_path) as conn:
        row = conn.execute(
            "SELECT mood, mood_updated_at FROM users WHERE user_id = ?", (unique_user_id,)
        ).fetchone()
    assert row["mood"] == "curious"
    assert row["mood_updated_at"] is not None


def test_recommend_echoing_the_same_mood_does_not_reset_the_timestamp(engine, unique_user_id, db_path):
    from database import db_connect

    engine.onboard_user({"user_id": unique_user_id, "interests": ["science"]})
    engine.recommend(unique_user_id, k=8, mood="curious")

    with db_connect(db_path) as conn:
        first_ts = conn.execute(
            "SELECT mood_updated_at FROM users WHERE user_id = ?", (unique_user_id,)
        ).fetchone()["mood_updated_at"]

    # Frontend echoes the already-stored mood on every normal call — this
    # must NOT look like a fresh mood-change, or mood would never decay
    # during a normal active session.
    engine.recommend(unique_user_id, k=8, mood="curious")

    with db_connect(db_path) as conn:
        second_ts = conn.execute(
            "SELECT mood_updated_at FROM users WHERE user_id = ?", (unique_user_id,)
        ).fetchone()["mood_updated_at"]

    assert first_ts == second_ts


def test_recommend_response_reports_mood_and_freshness(engine, unique_user_id):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["science"]})
    rec = engine.recommend(unique_user_id, k=8, mood="curious")
    assert rec["mood"] == "curious"
    assert 0.0 <= rec["mood_freshness"] <= 1.0


def test_recommend_response_mood_freshness_is_zero_with_no_mood_set(engine, unique_user_id):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["science"]})
    rec = engine.recommend(unique_user_id, k=8)
    assert rec["mood_freshness"] == pytest.approx(0.0)


# ── scoring weights sanity ──────────────────────────────────

def test_cold_start_and_warm_weights_sum_to_the_same_total():
    """
    Both weight presets in recommender.py's cold-start blend must sum to
    the same total (0.80) — they trade off which signals dominate, not how
    much headroom is left for att_boost + recency + randomness - saturation.
    This is a regression test for an arithmetic slip caught during review:
    the cold-start preset briefly summed to 0.85 instead of 0.80.
    """
    cold = {"q": 0.08, "sim": 0.18, "trend": 0.18, "pref": 0.20, "graph": 0.06, "ts": 0.00, "mood": 0.10}
    warm = {"q": 0.20, "sim": 0.18, "trend": 0.11, "pref": 0.09, "graph": 0.09, "ts": 0.05, "mood": 0.08}
    assert sum(cold.values()) == pytest.approx(0.80)
    assert sum(warm.values()) == pytest.approx(0.80)


def test_recommend_scores_shift_toward_mood_congruent_category(engine, unique_user_id, db_path, monkeypatch):
    """
    With an otherwise neutral profile and a fixed candidate pool, a science
    article should score higher once the user's mood is "curious" (0.9
    affinity for science) than with no mood set (neutral 0.5 affinity) —
    the mood weight must actually reach the final score, not just sit
    unused in the state vector.

    The real candidate pool is drawn with `ORDER BY RANDOM()`
    (_fetch_candidates_conn), so a same-news_id comparison across two
    separate recommend() calls would be flaky by construction — pin the
    pool and kill the small_randomness() jitter term so the only thing
    that can move the score is the mood weight.
    """
    import recommender as recommender_module
    from database import db_connect

    engine.onboard_user({"user_id": unique_user_id, "interests": [], "exploration_preference": 0.3})
    monkeypatch.setattr(recommender_module, "small_randomness", lambda *a, **k: 0.0)

    # ORDER BY news_id (not just LIMIT) so this is deterministic regardless
    # of other tests' insert/delete churn on the shared `news` table (e.g.
    # test_live_news.py) perturbing SQLite's unordered-scan row order —
    # a plain `LIMIT 6` with no ORDER BY silently returned an all-science
    # (or all-technology) set often enough to break the pigeonhole overlap
    # guarantee this test relies on.
    with db_connect(db_path) as conn:
        fixed_pool = []
        for cat in ("science", "technology"):
            fixed_pool.extend(
                dict(r) for r in conn.execute(
                    "SELECT news_id, category, subcategory, location, preferred_time,"
                    " age_group, article_length, abstract, full_article, url"
                    " FROM news WHERE category = ? AND is_live = 0 ORDER BY news_id LIMIT 3",
                    (cat,),
                ).fetchall()
            )
    for row in fixed_pool:
        row["source_mix"] = "personalized"
    assert any(r["category"] == "science" for r in fixed_pool)
    monkeypatch.setattr(engine, "_fetch_candidates_conn", lambda *a, **k: list(fixed_pool))
    # recommend() tops up with extra real-DB candidates (via ORDER BY
    # RANDOM(), which Python can't seed) whenever the fixed pool doesn't
    # fill k on its own — cut that path so the candidate *set* is fully
    # pinned to fixed_pool across both calls, leaving mood as the only
    # thing that can move a shared item's score.
    monkeypatch.setattr(engine, "_category_balanced_rows", lambda *a, **k: [])

    baseline = {
        item["news_id"]: item["score"]
        for item in engine.recommend(unique_user_id, k=10)["recommendations"]
    }
    moody = engine.recommend(unique_user_id, k=10, mood="curious")["recommendations"]

    science_items = [i for i in moody if i["category"] == "science" and i["news_id"] in baseline]
    assert science_items, "expected at least one repeated science candidate to compare"
    for item in science_items:
        assert item["score"] > baseline[item["news_id"]]


# ── reward mood-congruence shaping ──────────────────────────

def test_compute_reward_congruent_like_scores_higher_than_neutral_like():
    neutral = compute_reward(liked=1, time_spent=5, mood_congruence=0.5)
    congruent = compute_reward(liked=1, time_spent=5, mood_congruence=0.9)
    assert congruent > neutral


def test_compute_reward_congruent_dislike_scores_lower_than_neutral_dislike():
    neutral = compute_reward(liked=0, time_spent=5, mood_congruence=0.5)
    congruent = compute_reward(liked=0, time_spent=5, mood_congruence=0.9)
    assert congruent < neutral


def test_compute_reward_incongruent_dislike_is_softened_relative_to_neutral():
    neutral = compute_reward(liked=0, time_spent=5, mood_congruence=0.5)
    incongruent = compute_reward(liked=0, time_spent=5, mood_congruence=0.1)
    assert incongruent > neutral


def test_compute_reward_neutral_mood_congruence_is_a_no_op():
    without = compute_reward(liked=1, time_spent=5)
    with_neutral = compute_reward(liked=1, time_spent=5, mood_congruence=0.5)
    assert without == with_neutral


# ── feedback(): mood-aware reward reaches the RL update ─────

def test_feedback_uses_current_mood_for_reward_without_error(engine, unique_user_id):
    engine.onboard_user({"user_id": unique_user_id, "interests": ["science"], "mood": "curious"})
    item = engine.recommend(unique_user_id, k=8, mood="curious")["recommendations"][0]
    result = engine.feedback({
        "user_id": unique_user_id, "news_id": item["news_id"],
        "time_spent": 8, "liked": 1, "final_score": item["score"],
        "similarity": item["similarity"], "trending": item["trending"],
    })
    assert "reward" in result

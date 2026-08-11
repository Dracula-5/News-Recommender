"""
News recommendation engine.

Key changes vs. original:
  * EnhancedNLPEngine (BM25 + TF-IDF + LSA) replaces the bare TF-IDF vectorizer
  * UserInterestGraph (time-decayed knowledge graph) feeds graph_score into state
  * State expanded to 10-D: adds graph_score, ts_reward, time_sin/cos, session_like_rate
  * DDQNAgent now uses HybridPolicy (TS + UCB + Q-blend) for category selection
  * graph_memory.save_to_db() called on every feedback to persist the graph
"""
from __future__ import annotations

import logging
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from database import DB_PATH, db_connect, fetch_categories, import_csvs, init_db
from ddqn_agent import DDQNAgent
from graph_memory import UserInterestGraph
from nlp_engine import EnhancedNLPEngine
from utils import (
    category_preference_from_text,
    clamp,
    compute_interaction_score,
    compute_reward,
    diversity_limited,
    lexical_similarity,
    normalize_0_1,
    small_randomness,
)
from webcam_attention import (
    compute_attention_score,
    compute_energy_score,
    compute_gaze_score,
    get_latest_attention_snapshot,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _time_features() -> tuple[float, float]:
    """Return (sin, cos) of the current hour mapped to [0, 1]."""
    h = datetime.now().hour
    sin_ = (1.0 + math.sin(2 * math.pi * h / 24)) / 2
    cos_ = (1.0 + math.cos(2 * math.pi * h / 24)) / 2
    return sin_, cos_


class NewsRecommender:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._agents: dict[str, DDQNAgent] = {}
        self._nlp = EnhancedNLPEngine(n_lsa=64)
        self._graph = UserInterestGraph(db_path)
        self._nlp_fitted = False
        init_db(db_path)
        with db_connect(db_path) as conn:
            news_count = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        if news_count == 0:
            import_csvs(db_path=db_path, force=False)

    # ──────────────────────────────────────────────────────
    #  Category helpers
    # ──────────────────────────────────────────────────────

    def categories(self) -> list[str]:
        return fetch_categories(self.db_path)

    def _agent(self, user_id: str) -> DDQNAgent:
        cats = self.categories()
        agent = self._agents.get(user_id)
        if agent is None or agent.categories != cats:
            agent = DDQNAgent(cats, user_id=user_id, db_path=self.db_path)
            self._agents[user_id] = agent
        return agent

    # ──────────────────────────────────────────────────────
    #  NLP engine initialisation
    # ──────────────────────────────────────────────────────

    def _ensure_nlp(self) -> None:
        if self._nlp_fitted:
            return
        try:
            with db_connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT news_id, category, subcategory, abstract FROM news LIMIT 20000"
                ).fetchall()
            corpus  = [f"{r['category']} {r['subcategory']} {r['abstract']}" for r in rows]
            doc_ids = [r["news_id"] for r in rows]
            self._nlp.fit(corpus, doc_ids)
            self._nlp_fitted = True
        except Exception as exc:
            logger.warning("NLP engine init failed: %s", exc)

    # ──────────────────────────────────────────────────────
    #  State builder  (10-D)
    # ──────────────────────────────────────────────────────

    def build_state_from_row(
        self,
        user_row,
        category: str,
        all_prefs: dict[str, float],
        overrides: dict[str, Any] | None = None,
    ) -> list[float]:
        """
        10-dimensional state vector:
          [0] attention_score       — webcam / frontend attention
          [1] interaction_score     — dwell + clicks + poll
          [2] recent_activity       — smoothed recent engagement
          [3] category_pref         — per-category preference [0,1]
          [4] exploration_signal    — exploration tendency
          [5] graph_score           — knowledge-graph interest for category
          [6] ts_reward             — Thompson Sampling expected reward
          [7] time_sin              — cyclical hour encoding (sin)
          [8] time_cos              — cyclical hour encoding (cos)
          [9] session_like_rate     — recent like fraction
        """
        ov = overrides or {}
        time_sin, time_cos = _time_features()
        return [
            normalize_0_1(ov.get("attention_score",    user_row["attention_score"])),
            normalize_0_1(ov.get("interaction_score",  user_row["interaction_score"])),
            normalize_0_1(user_row["recent_activity"]),
            float(all_prefs.get(category, 0.2)),
            normalize_0_1(ov.get("exploration_signal", user_row["exploration_signal"])),
            float(ov.get("graph_scores", {}).get(category, 0.5)),
            float(ov.get("ts_rewards",   {}).get(category, 0.5)),
            float(ov.get("time_sin", time_sin)),
            float(ov.get("time_cos", time_cos)),
            float(ov.get("session_like_rate", 0.5)),
        ]

    def build_state(self, user_id: str, category: str, overrides: dict | None = None) -> list[float]:
        row       = self._user_row(user_id)
        all_prefs = self._all_category_prefs(user_id)
        return self.build_state_from_row(row, category, all_prefs, overrides)

    # ──────────────────────────────────────────────────────
    #  DB helpers
    # ──────────────────────────────────────────────────────

    def _user_row(self, user_id: str):
        with db_connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                self.onboard_user({"user_id": user_id, "interests": [], "exploration_preference": 0.3})
                row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row

    def _all_category_prefs(self, user_id: str) -> dict[str, float]:
        with db_connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT category, preference FROM user_category_prefs WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return {r["category"]: clamp(r["preference"]) for r in rows}

    def blocked_categories(self, user_id: str, agent_step: int) -> set[str]:
        with db_connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT category FROM user_category_prefs "
                "WHERE user_id = ? AND dislikes >= 2 AND blocked_until_step > ?",
                (user_id, agent_step),
            ).fetchall()
        return {r["category"] for r in rows}

    def _recent_memory(self, user_id: str) -> tuple[set[str], Counter]:
        with db_connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT news_id, category FROM user_memory "
                "WHERE user_id = ? ORDER BY position DESC LIMIT 80",
                (user_id,),
            ).fetchall()
        return {r["news_id"] for r in rows}, Counter(r["category"] for r in rows)

    # ──────────────────────────────────────────────────────
    #  Onboarding
    # ──────────────────────────────────────────────────────

    def onboard_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        categories = self.categories()
        user_id    = payload.get("user_id") or f"user_{random.randint(10000, 99999)}"
        interests  = payload.get("interests", [])
        if isinstance(interests, str):
            interests = [interests]
        interest_text = " ".join(interests + [payload.get("mood", ""), payload.get("sample_click", "")])
        exploration   = float(payload.get("exploration_preference", 0.25))
        attention     = 0.55 + min(float(payload.get("time_available", 10)) / 60.0, 0.25)
        interaction   = 0.45 + (0.15 if payload.get("sample_click") else 0.0)

        with db_connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users
                (user_id, user_interest_text, mood, time_available, time_of_day,
                 current_location, attention_score, interaction_score, recent_activity,
                 exploration_signal, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    user_id, interest_text,
                    payload.get("mood", ""),
                    float(payload.get("time_available", 10)),
                    payload.get("time_of_day", ""),
                    payload.get("location", ""),
                    clamp(attention), clamp(interaction), clamp(interaction), clamp(exploration),
                ),
            )
            prefs = category_preference_from_text(categories, interest_text)
            for cat, pref in prefs.items():
                if cat in categories:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO user_category_prefs
                        (user_id, category, preference, dislikes, blocked_until_step, updated_at)
                        VALUES (?, ?, ?, 0, 0, CURRENT_TIMESTAMP)
                        """,
                        (user_id, cat, pref),
                    )
        self._agent(user_id).save()
        return {"user_id": user_id, "categories": categories, "message": "onboarded"}

    # ──────────────────────────────────────────────────────
    #  Candidate retrieval (same 3-pool mix as original)
    # ──────────────────────────────────────────────────────

    def _fetch_candidates_conn(
        self,
        conn,
        user_id:    str,
        seen_news:  set[str],
        blocked:    set[str],
        pool_size:  int = 25,
        location:   str | None = None,
        force_mode: str | None = None,   # "trending" → skip personalised pool
    ) -> list[dict[str, Any]]:
        blocked_sql = ",".join("?" for _ in blocked) or "''"
        seen_sql    = ",".join("?" for _ in seen_news) or "''"
        base_filter = f"news.category NOT IN ({blocked_sql}) AND news.news_id NOT IN ({seen_sql})"
        base_params = list(blocked) + list(seen_news)
        if location:
            base_filter += " AND news.location = ?"
            base_params.append(location)

        COLS = (
            "news.news_id, news.category, news.subcategory, news.location,"
            " news.preferred_time, news.age_group, news.article_length,"
            " news.abstract, news.full_article, news.url"
        )

        # Pool composition: normal vs. force-trending
        if force_mode == "trending":
            mix = [
                ("trending", max(1, round(pool_size * 0.80))),
                ("random",   pool_size - round(pool_size * 0.80)),
            ]
        else:
            mix = [
                ("personalized", max(1, round(pool_size * 0.60))),
                ("trending",     max(1, round(pool_size * 0.20))),
                ("random",       pool_size - round(pool_size * 0.60) - round(pool_size * 0.20)),
            ]

        candidates: list[dict] = []
        global_seen = set(seen_news)

        for pool_mode, count in mix:
            if pool_mode == "personalized":
                sql = (
                    f"SELECT {COLS} FROM news"
                    " LEFT JOIN user_category_prefs ucp"
                    "   ON ucp.user_id = ? AND ucp.category = news.category"
                    f" WHERE {base_filter}"
                    " ORDER BY COALESCE(ucp.preference, 0.2) DESC, RANDOM()"
                    " LIMIT ?"
                )
                params = [user_id] + base_params + [max(25, count * 3)]
            elif pool_mode == "trending":
                sql = (
                    f"SELECT {COLS} FROM news"
                    " LEFT JOIN category_stats cs ON cs.category = news.category"
                    f" WHERE {base_filter}"
                    " ORDER BY COALESCE(cs.trending_score, 0) DESC, RANDOM()"
                    " LIMIT ?"
                )
                params = base_params + [max(25, count * 3)]
            else:
                sql = f"SELECT {COLS} FROM news WHERE {base_filter} ORDER BY RANDOM() LIMIT ?"
                params = base_params + [max(25, count * 3)]

            added = 0
            for r in conn.execute(sql, params).fetchall():
                row = dict(r)
                if row["news_id"] not in global_seen:
                    row["source_mix"] = pool_mode
                    candidates.append(row)
                    global_seen.add(row["news_id"])
                    added += 1
                    if added >= count:
                        break

        return candidates

    def _category_balanced_rows(
        self,
        user_id:    str,
        per_category: int = 4,
        blocked:    set[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        blocked     = blocked     or set()
        exclude_ids = exclude_ids or set()
        seen_news, _ = self._recent_memory(user_id)
        exclude = seen_news | exclude_ids
        rows: list[dict] = []
        with db_connect(self.db_path) as conn:
            for cat in self.categories():
                if cat in blocked:
                    continue
                exc_sql = ",".join("?" for _ in exclude) or "''"
                for row in conn.execute(
                    f"SELECT news_id, category, subcategory, location, preferred_time,"
                    f" age_group, article_length, abstract, full_article, url"
                    f" FROM news WHERE category = ? AND news_id NOT IN ({exc_sql})"
                    f" ORDER BY RANDOM() LIMIT ?",
                    [cat, *exclude, per_category],
                ).fetchall():
                    rows.append(dict(row))
                    exclude.add(row["news_id"])
        return rows

    # ──────────────────────────────────────────────────────
    #  Similarity (uses enhanced NLP engine)
    # ──────────────────────────────────────────────────────

    def _similarities(
        self,
        user_text:  str,
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        if not candidates:
            return []
        docs = [f"{c.get('category','')} {c.get('subcategory','')} {c.get('abstract','')}" for c in candidates]
        nids = [c.get("news_id", f"__tmp_{i}") for i, c in enumerate(candidates)]

        # Ensure NLP engine is fitted before trying it
        self._ensure_nlp()

        # Try the enhanced NLP engine first
        if self._nlp_fitted:
            try:
                scores_map = self._nlp.score_candidates(user_text, nids)
                return [float(scores_map.get(nid, 0.0)) for nid in nids]
            except Exception as exc:
                logger.warning("NLP score_candidates failed: %s", exc)

        # Fallback: fit a quick TF-IDF on the small candidate set
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vecs = TfidfVectorizer(max_features=3000, stop_words="english").fit_transform(
                [user_text] + docs
            )
            sims = cosine_similarity(vecs[0:1], vecs[1:]).ravel()
            return [float(x) for x in sims]
        except Exception:
            return lexical_similarity(user_text, docs)

    # ──────────────────────────────────────────────────────
    #  Recommend
    # ──────────────────────────────────────────────────────

    def recommend(
        self,
        user_id:  str,
        k:        int = 8,
        mode:     str | None = None,
        location: str | None = None,
        mood:     str | None = None,
    ) -> dict[str, Any]:
        k          = max(8, min(10, int(k)))
        categories = self.categories()
        agent      = self._agent(user_id)

        # ── single DB round-trip ──────────────────────────
        with db_connect(self.db_path) as conn:
            user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if user is None:
                self.onboard_user({"user_id": user_id, "interests": [], "exploration_preference": 0.3})
                user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()

            blocked = {
                r["category"]
                for r in conn.execute(
                    "SELECT category FROM user_category_prefs "
                    "WHERE user_id = ? AND dislikes >= 2 AND blocked_until_step > ?",
                    (user_id, agent.step),
                ).fetchall()
            }
            memory_rows = conn.execute(
                "SELECT news_id, category FROM user_memory WHERE user_id = ? "
                "ORDER BY position DESC LIMIT 80",
                (user_id,),
            ).fetchall()
            seen_news              = {r["news_id"] for r in memory_rows}
            recent_category_counts = Counter(r["category"] for r in memory_rows)

            liked_rows = conn.execute(
                """
                SELECT category, COUNT(*) AS likes
                FROM (SELECT category FROM interactions WHERE user_id = ? AND liked = 1
                      ORDER BY created_at DESC LIMIT 30)
                GROUP BY category
                """,
                (user_id,),
            ).fetchall()
            recent_liked_boosts = {r["category"]: min(0.20, 0.05 * int(r["likes"])) for r in liked_rows}

            all_prefs = {
                r["category"]: clamp(r["preference"])
                for r in conn.execute(
                    "SELECT category, preference FROM user_category_prefs WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            }
            trending = {
                r["category"]: float(r["trending_score"] or 0)
                for r in conn.execute("SELECT category, trending_score FROM category_stats")
            }
            recent_rows = conn.execute(
                "SELECT category, liked FROM interactions WHERE user_id = ? "
                "ORDER BY rowid DESC LIMIT 20",
                (user_id,),
            ).fetchall()
            liked_cats    = {r["category"] for r in recent_rows if r["liked"] == 1}
            disliked_cats = {r["category"] for r in recent_rows if r["liked"] == 0}

            candidates = self._fetch_candidates_conn(
                conn, user_id, seen_news, blocked, pool_size=25,
                location=location, force_mode=mode,
            )

        # ── attention ─────────────────────────────────────
        snap      = get_latest_attention_snapshot()
        _norm_att = snap.get("normalized_attention")
        if _norm_att is None or not isinstance(_norm_att, (int, float)):
            _raw      = float(snap.get("attention_score") or 0.0)
            _norm_att = _raw if _raw <= 1.0 else _raw / 100.0
        current_attention = clamp(float(_norm_att))
        attention_source  = str(snap.get("source", "frontend") or "frontend")

        # ── session like rate ─────────────────────────────
        total_r   = len(recent_rows)
        liked_r   = sum(1 for r in recent_rows if r["liked"] == 1)
        session_like_rate = liked_r / max(1, total_r)

        # ── graph scores + TS rewards ─────────────────────
        cat_indices  = {cat: idx for idx, cat in enumerate(categories)}
        graph_scores = {cat: self._graph.get_category_score(user_id, cat) for cat in categories}
        ts_rewards   = dict(zip(agent.categories, agent.hybrid.ts.get_expected_reward()))
        time_sin, time_cos = _time_features()

        state_overrides = {
            "attention_score":   current_attention,
            "graph_scores":      graph_scores,
            "ts_rewards":        ts_rewards,
            "time_sin":          time_sin,
            "time_cos":          time_cos,
            "session_like_rate": session_like_rate,
        }

        # ── scoring inner function ────────────────────────
        def score_candidates(rows: list[dict], lc: set, dc: set) -> list[dict]:
            if not rows:
                return []
            states   = [self.build_state_from_row(user, r["category"], all_prefs, state_overrides) for r in rows]
            q_matrix = agent.q_values_batch(states)
            sims     = self._similarities(user["user_interest_text"], rows)

            out = []
            for row, sim, q_row in zip(rows, sims, q_matrix):
                cat     = row["category"]
                q_raw   = float(q_row[cat_indices[cat]]) if cat in cat_indices else 0.0
                q_norm  = float(1.0 / (1.0 + math.exp(-q_raw)))
                trend   = clamp(trending.get(cat, 0.0))
                recency = recent_liked_boosts.get(cat, 0.0)
                saturation = min(0.35, 0.10 * recent_category_counts.get(cat, 0))
                cat_pref   = all_prefs.get(cat, 0.2)
                g_score    = graph_scores.get(cat, 0.5)
                ts_rwd     = ts_rewards.get(cat, 0.5)
                att_boost  = 0.20 * current_attention * (1.2 if attention_source == "webcam" else 1.0)

                # Core scoring:  DDQN + NLP sim + trending + preference + graph + TS + attention
                final = clamp(
                    0.22 * q_norm
                    + 0.20 * clamp(sim)
                    + 0.12 * trend
                    + 0.10 * cat_pref
                    + 0.10 * g_score        # graph memory contribution
                    + 0.06 * ts_rwd         # Thompson Sampling confidence
                    + att_boost
                    + recency
                    + small_randomness()
                    - saturation
                )
                if cat in lc:
                    final = clamp(final + 0.20)
                if cat in dc:
                    final = clamp(final - 0.50)

                row.update({
                    "score":             round(final,         4),
                    "q_value":           round(q_raw,         4),
                    "similarity":        round(float(sim),    4),
                    "trending":          round(trend,         4),
                    "graph_score":       round(g_score,       4),
                    "ts_reward":         round(ts_rwd,        4),
                    "recency_boost":     round(recency,       4),
                    "saturation_penalty": round(saturation,   4),
                    "hitl_decision": (
                        "auto_skip"    if final < 0.03 else
                        "ask_continue" if final < 0.50 else
                        "keep"
                    ),
                })
                out.append(row)
            return out

        scored = score_candidates(candidates, liked_cats, disliked_cats)
        scored.sort(key=lambda x: x["score"], reverse=True)

        liked_pool:   list[dict] = []
        trending_pool: list[dict] = []
        random_pool:  list[dict] = []

        for item in scored:
            cat = item["category"]
            if cat in disliked_cats:
                continue
            if cat in liked_cats:
                liked_pool.append(item)
            elif item.get("trending", 0) > 0.5:
                trending_pool.append(item)
            else:
                random_pool.append(item)

        random.shuffle(liked_pool)
        random.shuffle(trending_pool)
        random.shuffle(random_pool)

        liked_k    = int(0.45 * k)
        trending_k = int(0.25 * k)
        random_k   = k - liked_k - trending_k

        final_mix: list[dict] = []
        final_mix.extend(liked_pool[:liked_k])
        final_mix.extend(trending_pool[:trending_k])
        final_mix.extend(random_pool[:random_k])

        remaining = k - len(final_mix)
        if remaining > 0:
            leftovers = liked_pool[liked_k:] + trending_pool[trending_k:] + random_pool[random_k:]
            random.shuffle(leftovers)
            final_mix.extend(leftovers[:remaining])

        if len(final_mix) < k:
            fallback = self._category_balanced_rows(user_id, per_category=2, blocked=blocked, exclude_ids=seen_news)
            extra = score_candidates([{**r, "source_mix": "balanced"} for r in fallback], liked_cats, disliked_cats)
            for item in extra:
                if item["category"] not in disliked_cats:
                    final_mix.append(item)
            random.shuffle(final_mix)

        cat_count: dict[str, int] = {}
        filtered: list[dict] = []
        for item in final_mix:
            cat = item["category"]
            if cat_count.get(cat, 0) < 2:
                filtered.append(item)
                cat_count[cat] = cat_count.get(cat, 0) + 1

        selected = filtered[:k]
        random.shuffle(selected)

        logger.info(
            "recommend user=%s k=%s selected=%s",
            user_id, k,
            [(i["news_id"], i["category"], i["score"]) for i in selected],
        )
        return {
            "user_id":              user_id,
            "k":                    k,
            "blocked_categories":   sorted(blocked),
            "recommendations":      selected,
        }

    # ──────────────────────────────────────────────────────
    #  Poll feedback
    # ──────────────────────────────────────────────────────

    def poll_feedback(self, user_id: str, news_id: str, liked: int) -> dict[str, Any]:
        with db_connect(self.db_path) as conn:
            news = conn.execute("SELECT category FROM news WHERE news_id = ?", (news_id,)).fetchone()
            if news is None:
                raise ValueError(f"Unknown news_id: {news_id}")
            category = news["category"]
            row = conn.execute(
                "SELECT preference, dislikes FROM user_category_prefs "
                "WHERE user_id = ? AND category = ?",
                (user_id, category),
            ).fetchone()
            old_pref    = float(row["preference"]) if row else 0.2
            old_dislikes = int(row["dislikes"])    if row else 0
            pref_delta  = 0.10 if liked else -0.12
            new_pref    = clamp(old_pref + pref_delta)
            dislikes    = 0 if liked else old_dislikes + 1
            agent       = self._agent(user_id)
            blocked_until = (agent.step + 15) if (not liked and dislikes >= 2) else 0
            conn.execute(
                """
                INSERT OR REPLACE INTO user_category_prefs
                (user_id, category, preference, dislikes, blocked_until_step, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, category, new_pref, dislikes, blocked_until),
            )
        return {
            "user_id":   user_id,
            "news_id":   news_id,
            "category":  category,
            "liked":     liked,
            "pref_delta": pref_delta,
            "new_pref":  round(new_pref, 4),
            "message":   "poll feedback applied",
        }

    # ──────────────────────────────────────────────────────
    #  Feedback (main training loop)
    # ──────────────────────────────────────────────────────

    def feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id      = payload["user_id"]
        news_id      = payload["news_id"]
        time_spent   = float(payload.get("time_spent",   0))
        liked        = int(payload.get("liked",          0))
        skipped      = int(payload.get("skipped",        0))
        scroll_depth = float(payload.get("scroll_depth", 0.0))
        click_val    = float(payload.get("click_val",    1 if liked else 0))
        poll_val     = clamp(float(payload.get("poll_val",   0.5)))
        final_score  = float(payload.get("final_score",  0.0))
        similarity   = float(payload.get("similarity",   0.0))
        trending_val = float(payload.get("trending",     0.0))

        brightness   = self._metric(payload, "brightness",   50.0,   0.0, 100.0)
        eye_openness = self._metric(payload, "eye_openness", 70.0,   0.0, 100.0)
        movement     = self._metric(payload, "movement",     20.0,   0.0, 100.0)
        energy_score = self._metric(
            payload, "energy_score",
            compute_energy_score(brightness, eye_openness, movement),
            0.0, 100.0,
        )
        gaze_score = self._metric(
            payload, "gaze_score",
            compute_gaze_score(brightness, eye_openness, movement),
            0.0, 1.0,
        )
        normalized_attention = payload.get("normalized_attention", payload.get("attention_score", 0.5))
        normalized_attention = normalize_0_1(normalized_attention)
        if normalized_attention <= 0 and (energy_score > 0 or gaze_score > 0):
            normalized_attention = normalize_0_1(compute_attention_score(energy_score, gaze_score))
        attention_score  = normalized_attention
        face_detected    = int(bool(payload.get("face_detected", 0)))
        attention_source = str(payload.get("attention_source", payload.get("source", "frontend")) or "frontend")

        with db_connect(self.db_path) as conn:
            news = conn.execute("SELECT category FROM news WHERE news_id = ?", (news_id,)).fetchone()
            if news is None:
                raise ValueError(f"Unknown news_id: {news_id}")
            category = news["category"]
            user_row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            all_prefs = {
                r["category"]: clamp(r["preference"])
                for r in conn.execute(
                    "SELECT category, preference FROM user_category_prefs WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            }

        long_term_history = clamp(all_prefs.get(category, 0.2))
        interaction_score = compute_interaction_score(click_val, poll_val, time_spent, long_term_history)
        reward            = compute_reward(liked, time_spent, similarity, scroll_depth)

        # ── update graph memory ───────────────────────────
        self._graph.record_interaction(user_id, category, reward, bool(liked), time_spent)

        # ── build state ───────────────────────────────────
        graph_scores = {cat: self._graph.get_category_score(user_id, cat) for cat in self.categories()}
        agent        = self._agent(user_id)
        ts_rewards   = dict(zip(agent.categories, agent.hybrid.ts.get_expected_reward()))
        recent_rows  = []
        with db_connect(self.db_path) as conn:
            recent_rows = conn.execute(
                "SELECT liked FROM interactions WHERE user_id = ? ORDER BY rowid DESC LIMIT 20",
                (user_id,),
            ).fetchall()
        session_like_rate = sum(1 for r in recent_rows if r["liked"] == 1) / max(1, len(recent_rows))
        time_sin, time_cos = _time_features()

        state_overrides = {
            "attention_score":   attention_score,
            "interaction_score": interaction_score,
            "graph_scores":      graph_scores,
            "ts_rewards":        ts_rewards,
            "time_sin":          time_sin,
            "time_cos":          time_cos,
            "session_like_rate": session_like_rate,
        }
        state = self.build_state_from_row(user_row, category, all_prefs, state_overrides)

        categories = self.categories()
        q_value    = float(agent.q_values(state)[categories.index(category)]) if category in categories else 0.0

        pref_delta = 0.14 + 0.08 * attention_score if liked else -0.18 - 0.10 * (1.0 - attention_score)
        if time_spent > 6:
            pref_delta += 0.05
        elif time_spent < 2:
            pref_delta -= 0.05
        if time_spent > 15:
            pref_delta += 0.08 if liked else -0.08

        with db_connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT preference, dislikes FROM user_category_prefs "
                "WHERE user_id = ? AND category = ?",
                (user_id, category),
            ).fetchone()
            old_pref    = float(row["preference"]) if row else 0.2
            old_dislikes = int(row["dislikes"])    if row else 0
            new_pref    = clamp(old_pref + pref_delta)
            dislikes    = 0 if liked else old_dislikes + 1
            if not liked and attention_score < 0.35:
                blocked_until = agent.step + 30
            elif dislikes >= 2:
                blocked_until = agent.step + 20
            else:
                blocked_until = 0

            conn.execute(
                """
                INSERT OR REPLACE INTO user_category_prefs
                (user_id, category, preference, dislikes, blocked_until_step, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, category, new_pref, dislikes, blocked_until),
            )
            conn.execute(
                """
                INSERT INTO interactions
                (user_id, news_id, category, time_spent, liked, skipped, scroll_depth,
                 click_val, reward, attention_score, interaction_score, final_score,
                 q_value, similarity, trending, brightness, eye_openness, movement,
                 energy_score, gaze_score, normalized_attention, attention_source,
                 poll_val, long_term_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, news_id, category, time_spent, liked, skipped, scroll_depth,
                    click_val, reward, attention_score, interaction_score, final_score,
                    q_value, similarity, trending_val, brightness, eye_openness, movement,
                    energy_score, gaze_score, normalized_attention, attention_source,
                    poll_val, long_term_history,
                ),
            )
            conn.execute(
                """
                INSERT INTO attention_events
                (user_id, news_id, brightness, eye_openness, movement, energy_score,
                 gaze_score, attention_score, normalized_attention, face_detected, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, news_id, brightness, eye_openness, movement,
                    energy_score, gaze_score, attention_score, normalized_attention,
                    face_detected, attention_source,
                ),
            )
            stats = conn.execute(
                "SELECT impressions, clicks, likes, avg_time FROM category_stats WHERE category = ?",
                (category,),
            ).fetchone()
            impressions = int(stats["impressions"] if stats else 0) + 1
            clicks      = int(stats["clicks"]      if stats else 0) + int(click_val > 0)
            likes       = int(stats["likes"]       if stats else 0) + liked
            avg_time    = (
                (float(stats["avg_time"] if stats else 0) * (impressions - 1)) + time_spent
            ) / impressions
            log_denom     = max(math.log1p(impressions + 1), 1.0)
            trending_score = clamp(
                0.45 * (math.log1p(clicks) / log_denom)
                + 0.25 * (math.log1p(likes)  / log_denom)
                + 0.15 * min(avg_time / 20.0, 1.0)
                + 0.15 * attention_score
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO category_stats
                (category, impressions, clicks, likes, avg_time, trending_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (category, impressions, clicks, likes, avg_time, trending_score),
            )
            cur = conn.execute(
                "SELECT attention_score, interaction_score, exploration_signal FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            old_att = float(cur["attention_score"])    if cur else attention_score
            old_int = float(cur["interaction_score"])  if cur else interaction_score
            old_exp = float(cur["exploration_signal"] or 0.2) if cur else 0.2
            smoothed_att = clamp(0.35 * old_att + 0.65 * attention_score)
            smoothed_int = clamp(0.35 * old_int + 0.65 * interaction_score)
            exp_delta    = -0.02 if (liked and time_spent > 6) else 0.02 if not liked else 0.0
            new_exp      = clamp(old_exp + exp_delta)
            conn.execute(
                """
                UPDATE users
                SET attention_score = ?, interaction_score = ?, recent_activity = ?,
                    exploration_signal = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (smoothed_att, smoothed_int,
                 clamp((smoothed_att + smoothed_int) / 2), new_exp, user_id),
            )
            pos = conn.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM user_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO user_memory(user_id, position, news_id, category) VALUES (?, ?, ?, ?)",
                (user_id, pos, news_id, category),
            )
            conn.execute(
                """
                DELETE FROM user_memory WHERE user_id = ? AND position <= (
                    SELECT COALESCE(MAX(position), 0) - 100 FROM user_memory WHERE user_id = ?
                )
                """,
                (user_id, user_id),
            )

        # ── RL update ─────────────────────────────────────
        # Recompute session_like_rate after the interaction has been inserted
        # so next_state reflects the current feedback signal.
        with db_connect(self.db_path) as _rc:
            _recent = _rc.execute(
                "SELECT liked FROM interactions WHERE user_id = ? ORDER BY rowid DESC LIMIT 20",
                (user_id,),
            ).fetchall()
        _next_overrides = dict(state_overrides)
        _next_overrides["session_like_rate"] = (
            sum(1 for r in _recent if r["liked"] == 1) / max(1, len(_recent))
        )
        next_state = self.build_state_from_row(user_row, category, all_prefs, _next_overrides)
        loss = agent.observe(state, category, reward, next_state, done=False, liked=bool(liked))

        # ── persist graph ─────────────────────────────────
        self._graph.save_to_db(user_id)

        logger.info(
            "feedback user=%s news=%s cat=%s reward=%.3f loss=%s",
            user_id, news_id, category, reward, loss,
        )
        return {
            "user_id":            user_id,
            "news_id":            news_id,
            "category":           category,
            "reward":             reward,
            "interaction_score":  interaction_score,
            "attention_score":    attention_score,
            "brightness":         brightness,
            "eye_openness":       eye_openness,
            "movement":           movement,
            "energy_score":       energy_score,
            "gaze_score":         gaze_score,
            "normalized_attention": normalized_attention,
            "graph_score":        round(self._graph.get_category_score(user_id, category), 4),
            "loss":               loss,
            "blocked":            dislikes >= 2,
            "message":            "feedback applied and model updated",
        }

    # ──────────────────────────────────────────────────────
    #  Manual model update
    # ──────────────────────────────────────────────────────

    def update_model(self, user_id: str) -> dict[str, Any]:
        agent = self._agent(user_id)
        loss  = agent.train_step()
        agent.save()
        return {
            "user_id":  user_id,
            "step":     agent.step,
            "epsilon":  agent.epsilon,
            "loss":     loss,
        }

    # ──────────────────────────────────────────────────────
    #  Misc helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _metric(payload: dict, key: str, default: float, lo: float, hi: float) -> float:
        try:
            v = float(payload.get(key, default))
        except (TypeError, ValueError):
            v = default
        if not np.isfinite(v):
            v = default
        return max(lo, min(hi, v))

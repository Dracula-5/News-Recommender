# NeuroFeed — an adaptive news feed with a real RL/bandit stack behind it

NeuroFeed is a from-scratch news recommender that re-ranks its feed after
**every single interaction** — like, dislike, dwell time, even webcam
attention — using a Dueling Double DQN blended with contextual bandits, a
time-decayed interest graph, and an ensemble retrieval engine. It's a
personal systems project, built to be read end-to-end: every non-obvious
design decision below is one I made deliberately and can defend.

The UI is deliberately quiet about all of this — it reads like a
newspaper live-blog, not a debugging console. The RL/bandit/attention
machinery keeps running underneath; none of it is surfaced as a dashboard.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser (editorial-style feed UI)                                  │
│    - webcam attention starts silently in the background             │
│    - polls /attention for a live score, never renders it             │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │ REST (FastAPI)
┌───────────────────────────────▼─────────────────────────────────────┐
│  backend/main.py                                                    │
│    observability middleware: rate limit → request log → metrics     │
│    TTL cache (categories/users) · JWT-ish token auth · /health       │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │
┌───────────────────────────────▼─────────────────────────────────────┐
│  recommender.py  — NewsRecommender.recommend()                      │
│                                                                      │
│   candidates ─▶ score ─▶ cold-start weight blend ─▶ pool mixing      │
│                  │           (few interactions?)      (liked /       │
│                  │                                     trending /    │
│                  │                                     explore)      │
│                  ▼                                        │         │
│         per-candidate score =                              ▼         │
│           DDQN Q-value        (ddqn_agent.py)      MMR diversity     │
│         + NLP similarity      (nlp_engine.py)       re-rank          │
│         + trending            (category_stats)     (reranking.py)   │
│         + category preference                              │         │
│         + graph interest      (graph_memory.py)             ▼         │
│         + Thompson Sampling   (rl_policies.py)      final k articles │
│         + attention boost     (webcam_attention.py)                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │
                          SQLite (WAL mode)
```

## Why this exists

Most "recommender system" portfolio projects stop at content-based
filtering with cosine similarity. This one implements the parts that
actually make a production recommender hard:

- an online-learning agent that updates *during* the session, not offline
- the cold-start problem (new user, untrained agent) as an explicit,
  testable code path — not just "well it's random at first"
- diversity re-ranking, because pure top-K-by-score degenerates into five
  near-duplicate stories the moment one category scores well
- an implicit, physical signal (webcam-derived attention) folded into the
  same reward function as explicit likes/dislikes
- an evaluation methodology that works despite the fact that this is an
  interactive system, not a static ranking problem you can replay logs
  against

## RecSys techniques implemented

| Technique | Where | What it does |
|---|---|---|
| Dueling Double DQN | `ddqn_agent.py` | Learns per-category Q-values from a 10-D state (attention, interaction score, recency, category preference, exploration signal, graph score, TS reward, cyclical time-of-day, session like-rate) |
| Thompson Sampling + UCB1 | `rl_policies.py` | `HybridPolicy` blends both bandits with the DDQN, so category selection doesn't rely on Q-values alone while they're still noisy |
| Time-decayed interest graph | `graph_memory.py` | NetworkX graph of user↔category edges with exponential decay (1-week half-life) — captures "used to like this, drifting away" without a full retrain |
| BM25 + TF-IDF + LSA ensemble | `nlp_engine.py` | 40/40/20 blended retrieval; LSA (64-dim truncated SVD) also doubles as the content-embedding space for diversity re-ranking |
| **Cold-start weight blending** | `recommender.py` | Below `NEUROFEED_COLD_START_INTERACTION_THRESHOLD` (default 5) logged interactions, scoring shifts weight off the still-untrained Q-value/TS-reward terms and onto declared interests + trending — the standard RecSys cold-start fallback, made explicit and inspectable (`is_cold_start` in the API response) rather than left as an implicit side effect of an untrained model |
| **MMR diversity re-ranking** | `reranking.py` | Carbonell & Goldstein's Maximal Marginal Relevance over LSA content vectors — trades a configurable amount of relevance (`NEUROFEED_MMR_LAMBDA`, default 0.75) for spread, so a batch isn't five variations on one story |
| **Graph-informed exploration** | `graph_memory.py`, `recommender.py` | Co-interest edges (which categories a user tends to like *together*, time-decayed like everything else in the graph) give a small score boost to related-but-not-yet-liked categories, steering exploration toward what the graph has learned this user tends to like — not just random/trending |
| **Discounted bandits** | `rl_policies.py` | Thompson Sampling and UCB1 both discount old evidence (configurable decay, default 0.995) — plain bandit math assumes a stationary reward distribution, but a user's taste for a category drifts over weeks; without this, a few hundred interactions make the posterior nearly immovable |
| **Reading history & memory summary** | `recommender.py` | `get_history()` — the full interactions log, paginated, not just the 100-entry seen-exclusion list. `get_memory_summary()` — a human-readable digest (top/rising/fading interests, related categories) of the same signal the scorer consumes as opaque numbers |
| HITL feedback loop | `recommender.py`, `webcam_attention.py` | Every like/dislike/poll/dwell-time/attention-score event updates SQLite, category preferences, the replay buffer, and DDQN weights immediately — state/next_state are built to cleanly bracket each transition (not leak the event's own outcome into the "before" state) |
| Webcam attention | `webcam_attention.py` (OpenCV + MediaPipe FaceMesh) | Eye openness, gaze, head movement, brightness → a normalized attention score folded into the same reward signal as explicit feedback |

## Engineering practices (the "senior developer" half)

| Area | What's there |
|---|---|
| **Config** | `config.py` — one `pydantic-settings` source of truth (`.env.example`), replacing scattered `os.getenv()` calls and hardcoded constants |
| **Observability** | `logging_config.py` (structured, timestamped, ASCII-safe — see below), `metrics.py` (dependency-free Prometheus-format `/metrics`), request logging + latency histograms via middleware |
| **Caching** | `cache.py` — documented in-memory TTL cache with a Redis-shaped interface (`get`/`set`/`get_or_set`), wired into the hot, cheap-to-stale read paths |
| **Rate limiting** | `ratelimit.py` — fixed-window limiter, same "swap for Redis later" interface |
| **Evaluation** | `evaluation.py` — see below |
| **Authorization** | `authorize_user_id()` in `backend/main.py` — every per-user endpoint checks the bearer token's identity against the requested `user_id`; guest mode (no token) still works, a *mismatched* token gets 403. Constant-time password comparison (`hmac.compare_digest`), server-side logout (`POST /auth/logout` actually revokes the token, not just a client-side `localStorage.clear()`) |
| **Testing** | `tests/` — pytest suite (69 tests) against isolated, auto-seeded temp SQLite DBs (main + auth, separately); never touches the dev databases |
| **CI** | `.github/workflows/ci.yml` — seeds synthetic data, runs the full suite with coverage, builds the Docker image |
| **Containerization** | `Dockerfile` + `docker-compose.yml` — `docker compose up --build` and it's running, no manual data-loading step |
| **Reproducibility** | `scripts/seed_demo_data.py` — the original MIND-style dataset is proprietary/gitignored like any real production dataset would be, so this generates a small, topically-coherent synthetic stand-in (BM25/TF-IDF/LSA need real lexical signal to be worth demonstrating, so it's templated per-category, not random word soup) |

One concrete bug worth calling out, because it's the kind of thing that's
easy to gloss over in a portfolio writeup: while building the evaluation
harness below, the *first* run showed **precision degrading within a
session** instead of improving, which shouldn't happen. The root cause
was in the category-preference logic — a category could land in *both*
`liked_cats` and `disliked_cats` in the same scoring pass (binary "any
liked / any disliked in the last 20 interactions" meant a category the
user liked 90% of the time still picked up a stray miss as history
accumulated), so it ate a `+0.20` boost *and* a `-0.50` penalty at once —
net punishing a user's actual favorite category the more they used the
app. Switched to ratio-based, disjoint-by-construction classification
(`>=0.6` liked / `<=0.3` disliked) and re-ran the identical scenario:

| | Before | After |
|---|---|---|
| Precision@K | 0.464 | 0.623 |
| NDCG@K | 0.750 | 0.850 |
| Learning delta (late − early precision) | **−0.476** | **+0.343** |

That fix is locked in with a regression test
(`test_a_category_with_mostly_likes_is_not_also_classified_disliked`).

A second, more thorough bug-hunting pass turned up several more —
listing them because a portfolio project with zero bugs ever found in it
is a sign nobody looked hard enough, not a sign of quality:

- **Deadlock**: `graph_memory.get_co_interest_categories()` called
  another lock-acquiring method from inside its own lock
  (`threading.Lock` isn't reentrant). Latent — nothing called it yet —
  until this pass wired it into `recommend()`'s exploration boost, which
  would have made it a very real, very confusing hang.
- **Double-decay on every restart**: the interest graph re-applied decay
  for the same elapsed window twice after a process restart, because
  `last_time` wasn't rebased after the load-time decay was already
  applied.
- **Scores frozen between writes**: decay only ever ran on a write or a
  DB load — a category a user stopped engaging with kept its last score
  *forever* if the process just kept running, which is the opposite of
  "stale interests fade out." Now computed live at read time.
- **IDOR in the API**: despite the frontend sending `Authorization:
  Bearer <token>` on every request, nothing on the backend checked it
  against the request's `user_id` — any client could read or mutate
  *any* user's feed, profile, and feedback history. Found by writing a
  test for the fix, which is how the next one turned up too —
- **Test isolation gap**: `auth.py`'s DB path was a hardcoded constant
  with no override, so the isolated-test-DB setup only ever covered the
  main database — every test that touched signup/login had been quietly
  writing real accounts into the dev `auth.sqlite`.
- **Thread-per-user leak**: `DDQNAgent` spun up a dedicated background
  thread per user, cached for the process lifetime with no eviction.

Full writeups of each (root cause, fix, and the regression test that
locks it in) are in the commit messages — `git log` tells this part of
the story better than a paraphrase would.

## Evaluation methodology

Real logged interactions can't be replayed for offline evaluation here —
`recommend()` deliberately never re-shows an article once feedback has
been given on it, so "did we recommend what they later liked" is
circular by construction. Instead, `evaluation.py` runs a **simulated-user
episode** (the standard approach for evaluating interactive/bandit
recommenders): give a synthetic user a ground-truth set of preferred
categories, drive the *real* recommend → feedback loop against the real
engine and database for N rounds, and score each round against that
ground truth.

```bash
python evaluation.py --users 5 --rounds 15
# or: GET /eval/run?users=3&rounds=10&k=8
```

Reports Precision@K, NDCG@K, intra-list category diversity, catalog
coverage, and an early-vs-late precision split as a (noisy, small-sample,
honestly-reported) proxy for within-session learning.

## Quick start

### Docker (recommended — zero setup)

```bash
cd final
docker compose up --build
```

Open `http://localhost:8000/app`. The container seeds a synthetic demo
dataset on first boot (see [Reproducibility](#engineering-practices-the-senior-developer-half) above) — no dataset
download, no manual import step.

### Local

```bash
cd final
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/seed_demo_data.py --import   # one-time: generate + load demo data

python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app` for the UI, `/docs` for interactive API
docs.

### Running the tests

```bash
cd final
pip install -r requirements-dev.txt
pytest -v
```

## API overview

| Group | Endpoints |
|---|---|
| Core loop | `POST /onboard`, `POST /recommend`, `POST /feedback`, `POST /poll_feedback` |
| Auth | `POST /auth/signup`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/user/{id}`, `POST /auth/verify` |
| History & memory | `GET /history/{user_id}`, `GET /memory/summary/{user_id}` |
| Attention | `GET /attention`, `POST /attention/start`, `POST /attention/stop`, `GET /attention/stream` |
| RL / graph / NLP introspection | `GET /rl/metrics/{user_id}`, `GET /graph/stats/{user_id}`, `GET /graph/top/{user_id}`, `GET /graph/related/{user_id}/{category}`, `GET /nlp/status`, `POST /nlp/score` |
| Evaluation | `GET /eval/run` |
| Ops | `GET /health`, `GET /metrics`, `GET /categories`, `GET /users` |

Every per-user endpoint above (core loop, history/memory, profile) is
gated by `authorize_user_id()` — see [Authorization](#engineering-practices-the-senior-developer-half) above.

Full interactive docs at `/docs` once the server is running.

## Project layout

```
.
├── final/
│   ├── backend/main.py          FastAPI app, routing, observability middleware
│   ├── recommender.py           Candidate retrieval + scoring + feedback loop
│   ├── ddqn_agent.py            Dueling Double DQN
│   ├── rl_policies.py           Thompson Sampling, UCB1, HybridPolicy blend
│   ├── graph_memory.py          Time-decayed user↔category interest graph
│   ├── nlp_engine.py            BM25 + TF-IDF + LSA ensemble retrieval
│   ├── reranking.py             MMR diversity re-ranking
│   ├── evaluation.py            Simulated-user offline evaluation harness
│   ├── webcam_attention.py      OpenCV + MediaPipe attention pipeline
│   ├── auth.py                  Token-based auth (PBKDF2-HMAC-SHA256)
│   ├── database.py              SQLite schema + CSV import
│   ├── config.py                Centralized settings (pydantic-settings)
│   ├── cache.py / ratelimit.py / metrics.py / logging_config.py
│   ├── scripts/seed_demo_data.py  Synthetic dataset generator
│   ├── frontend/                 Editorial-style feed UI (vanilla HTML/CSS/JS)
│   ├── tests/                    pytest suite
│   ├── Dockerfile / docker-compose.yml
│   └── .github/workflows/ci.yml (at repo root)
```

## Known limitations

- The webcam attention pipeline needs a physical camera on the host; it's
  disabled by default in Docker (`NEUROFEED_ENABLE_WEBCAM_ATTENTION=false`)
  since containers don't get camera passthrough, and degrades gracefully
  to a "no face detected" state either way.
- The bundled dataset is synthetic (see above) — lexically coherent
  enough for BM25/TF-IDF/LSA to do real work, but not real news.
- The in-memory cache and rate limiter are single-process by design
  (documented, Redis-shaped interface); a multi-instance deployment would
  need to swap the backend, not the call sites.
- The early-vs-late learning-delta metric is noisy over short simulated
  episodes (small sample buckets) — treat it as a sanity check that the
  system *can* learn within a session, not a precise benchmark.
- `recommend()` prioritizes the category-diversity cap (max 2 per
  category) over hitting `k` exactly — on a thin or category-skewed
  candidate pool it can return fewer than `k` items rather than pad the
  batch with near-duplicate categories. Tried the other trade-off (relax
  the cap to hit the count target) and it produced worse batches, so this
  is deliberate, not an oversight.

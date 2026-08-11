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
| **Trending** | `recommender.py` | Site-wide (not personalized) top categories by `category_stats.trending_score` with a sample article each — the same trending signal the "trending" candidate pool already uses internally, surfaced directly for a always-something-to-read dashboard row |
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
| **Authorization** | `authorize_user_id()` in `backend/main.py` — every per-user endpoint checks the bearer token's identity against the requested `user_id`; guest mode (no token) still works, a *mismatched* token gets 403. `require_authorized_user_id()` is the stricter variant for endpoints with no legitimate guest use case (account-info lookups) — no token at all is rejected there, not let through |
| **Security hardening** | Constant-time password comparison (`hmac.compare_digest`); server-side logout (`POST /auth/logout` actually revokes the token); per-account login lockout after repeated failures (`login_throttle.py`, account-keyed, independent of the general per-IP limiter); a stricter rate limit on `/auth/signup` + `/auth/login` specifically; security response headers on every request (CSP with no `unsafe-inline`/`unsafe-eval` for scripts, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`) — see [Security notes](#security-notes) below |
| **Testing** | `tests/` — pytest suite (80 tests) against isolated, auto-seeded temp SQLite DBs (main + auth, separately); never touches the dev databases |
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

## Security notes

- **CSP with no `unsafe-inline`/`unsafe-eval` for scripts.** login.html and
  signup.html used to have their form-submit handlers as inline
  `<script>` blocks — moved to `login.js`/`signup.js` specifically so
  `script-src 'self'` could be strict. `style-src` does allow
  `'unsafe-inline'` (a few inline style attributes/properties — menu
  visibility, toast colors — and CSS-only injection can't execute script,
  a materially smaller risk than script-src would be); that's a
  deliberate trade-off, not an oversight.
- **Authorization, not just authentication.** Every per-user endpoint
  checks the bearer token's identity against the `user_id` in the
  request, not just whether *some* token was presented — see the IDOR
  entry in the bug list above for what this replaced.
- **Login brute-force lockout** (`login_throttle.py`) is intentionally
  account-keyed, not IP-keyed — closes the "guess slowly / rotate IPs"
  gap a per-IP rate limiter alone can't, at the cost of being griefable
  (fail a few logins for someone else's email, they're briefly locked
  out). A short default lockout (60s) bounds that; a production
  deployment fronted by something that can do CAPTCHA/2FA would do
  better, but "brute-forceable" beats "griefable" as the default when
  only one control is available.
- **Not done, and worth naming rather than glossing over:** auth tokens
  live in `localStorage`, which is readable by any script on the page —
  fine against network attackers, not against XSS. Moving to an
  `HttpOnly` cookie would close that, but changes the auth flow (CSRF
  now needs handling, since cookies are sent automatically) enough that
  it's a deliberate follow-up, not a small fix folded into this pass.

## Data & UX

- **`GET /trending`** — site-wide (not personalized) trending
  categories, cached the same way `/categories` is since it doesn't vary
  per user. Feeds the "Trending Now" row above the adaptive feed.
- **Settings panel** — interests, mood, location, exploration
  preference, and an attention-tracking on/off toggle, all editable
  after onboarding via `PATCH /user/{id}/interests` (now also accepts
  `location`) and the existing `/attention/start` \| `/attention/stop`.
  The toggle doesn't change the "starts silently by default" decision
  from the initial UI pass — it just gives the user a way to turn it
  back off, which a "quiet by default" UI still owes them.
- **Background prefetch** — once two articles are left in the current
  batch, the next one starts fetching in the background
  (`prefetchNextBatch()`), so the eventual refresh usually finds it
  already there instead of blocking on the network.
- **Sliding card transitions, toasts, a loading bar** — the feed
  animates the previous article out and the next one in on every
  advance (CSS transform/opacity, no JS animation library); like/dislike
  and settings-saved get a brief toast instead of nothing; a thin
  top-edge bar shows during the (now rarer, thanks to prefetch) times a
  batch has to load for real.

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

## Deploying to Render + Netlify

Splitting frontend and backend across two hosts is a different shape than
the Docker/local setup above — those serve the frontend *from* the
backend (`app.mount("/app", ...)` in `main.py`). Split across two
origins, the frontend needs to know the backend's URL and the backend
needs to allow the frontend's origin. `render.yaml` and `netlify.toml` at
the repo root pre-fill almost everything; two edits still need your
actual URLs, which don't exist until you've deployed once.

### 1. Backend on Render

**Via Blueprint (recommended)** — Render dashboard → **New +** → **Blueprint** →
connect this repo. It reads `render.yaml` and creates the service with:

| Field | Value |
|---|---|
| Root Directory | `final` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python scripts/seed_demo_data.py --import && python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

It'll prompt you for `NEUROFEED_CORS_ORIGINS` — leave it blank for now,
you don't have the Netlify URL yet (step 3 comes back to this).

**Via the dashboard manually** (no Blueprint) — same values as the table
above, entered by hand when creating a new Web Service, environment
`Python 3`. Either way, note the resulting URL
(`https://<your-service-name>.onrender.com`) — you need it in step 2.

A couple of things worth knowing before you deploy, not after:

- **Storage is ephemeral on Render's free tier** — `final/data/*.sqlite`
  and `final/models/*.pt` reset on every redeploy. The start command
  above reseeds the synthetic demo dataset automatically (same idempotent
  `import_csvs()` behavior the Docker entrypoint relies on — see
  `scripts/seed_demo_data.py`), so the app always comes up working, it
  just won't remember users/training across deploys. If you want that to
  persist, attach a Render **Disk**, mount it, and point
  `NEUROFEED_DB_PATH`/`NEUROFEED_AUTH_DB_PATH` at it — `MODEL_DIR` in
  `ddqn_agent.py` isn't yet settings-driven, so DDQN checkpoints
  specifically would still need that wired up as a follow-up.
- **Webcam attention won't run here regardless of settings** — Render
  has no camera device to open; `NEUROFEED_ENABLE_WEBCAM_ATTENTION=false`
  (already the default) is the only setting that makes sense for a cloud
  host. The app degrades to its normal "no face detected" fallback.
- **Free tier + PyTorch/OpenCV/MediaPipe** is a real resource squeeze
  (512MB RAM on Render's free instance type) — if the service crashes or
  the build times out, the Starter tier's extra memory is the first thing
  to try before debugging further.

### 2. Frontend on Netlify

**Via the dashboard** — Netlify → **Add new site** → **Import an existing
project** → connect this repo. It reads `netlify.toml` and needs nothing
filled in manually:

| Field | Value |
|---|---|
| Base directory | *(leave default — set via netlify.toml)* |
| Build command | *(none — static files, no build step)* |
| Publish directory | `final/frontend` |

Before or after the first deploy, make the two edits that depend on your
Render URL from step 1:

1. **`final/frontend/config.js`** — replace
   `https://YOUR-RENDER-SERVICE.onrender.com` with your actual Render URL.
2. **`netlify.toml`** — same replacement, inside the `connect-src` value
   of the `Content-Security-Policy` header (without it, the CSP blocks
   the frontend from calling the backend at all — this isn't optional).

Commit both, push, Netlify redeploys automatically. Note the resulting
Netlify URL (`https://<your-site-name>.netlify.app`) for step 3.

### 3. Close the loop: CORS

Back on Render, set the `NEUROFEED_CORS_ORIGINS` env var you left blank
in step 1 to your Netlify URL (exact origin, no trailing slash, e.g.
`https://your-site-name.netlify.app`), then redeploy the backend. Until
this is set, the backend still works standalone (`curl`, `/docs`,
Postman) — only browser requests *from the Netlify origin* are rejected
by CORS in the meantime.

### Sanity-checking the split deployment

```bash
curl https://<your-render-url>/health          # backend up, DB seeded
curl -I https://<your-netlify-url>/            # frontend serving, check headers
```

Then open the Netlify URL, sign up (or continue as guest), and confirm a
recommendation loads — that exercises frontend → backend → SQLite → NLP
engine end to end. If `/recommend` fails in the browser console with a
CORS or CSP error, it's almost always one of the two URL placeholders
above still unreplaced.

## API overview

| Group | Endpoints |
|---|---|
| Core loop | `POST /onboard`, `POST /recommend`, `POST /feedback`, `POST /poll_feedback` |
| Auth | `POST /auth/signup`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/user/{id}`, `POST /auth/verify` |
| History & memory | `GET /history/{user_id}`, `GET /memory/summary/{user_id}` |
| Attention | `GET /attention`, `POST /attention/start`, `POST /attention/stop`, `GET /attention/stream` |
| RL / graph / NLP introspection | `GET /rl/metrics/{user_id}`, `GET /graph/stats/{user_id}`, `GET /graph/top/{user_id}`, `GET /graph/related/{user_id}/{category}`, `GET /nlp/status`, `POST /nlp/score` |
| Evaluation | `GET /eval/run` |
| Ops | `GET /health`, `GET /metrics`, `GET /categories`, `GET /users`, `GET /trending` |

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
│   ├── login_throttle.py        Per-account login brute-force lockout
│   ├── database.py              SQLite schema + CSV import
│   ├── config.py                Centralized settings (pydantic-settings)
│   ├── cache.py / ratelimit.py / metrics.py / logging_config.py
│   ├── scripts/seed_demo_data.py  Synthetic dataset generator
│   ├── frontend/                 Editorial-style feed UI (vanilla HTML/CSS/JS)
│   │   └── config.js             Backend API base URL — the one line split deployment edits
│   ├── tests/                    pytest suite
│   └── Dockerfile / docker-compose.yml
├── render.yaml                   Render Blueprint (backend)
├── netlify.toml                  Netlify config (frontend)
└── .github/workflows/ci.yml
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

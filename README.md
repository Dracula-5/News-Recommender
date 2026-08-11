# NeuroFeed — AI-Powered Hyper-Personalized News Recommender

> A production-style news recommendation system powered by a **Dueling Double Deep Q-Network (DDQN)**, real-time **webcam attention tracking**, and a **poll-driven interaction scoring** model — all running locally with FastAPI + SQLite + a browser UI.

---

## What It Does

NeuroFeed learns what you want to read **in real time**. Every article interaction (like, dislike, dwell time, poll answer, eye-tracking data) feeds back into a reinforcement learning agent that reshapes your next recommendations immediately.

| Capability | Details |
|---|---|
| **RL Recommender** | Dueling DDQN selects categories; TF-IDF ranks articles within them |
| **Attention Tracking** | OpenCV + MediaPipe measures eye openness, gaze, head movement, brightness |
| **Interaction Score** | `0.55·click + 0.10·poll + 0.20·log_dwell + 0.15·history` |
| **Hyper-personalized Feed** | 45% liked categories · 25% trending · 30% exploration, reshuffled every 3 articles |
| **Like Boost** | Next 6 articles include 2–3 from the same liked category |
| **Dislike Suppression** | Disliked category limited to 1 article in next 5 |
| **Poll System** | After 20 s of reading, a popup asks "Do you like this?" — answer moderately adjusts category score |
| **JWT Auth** | Signup / login with hashed passwords and token-based sessions |
| **HITL Feedback Loop** | Every like / skip / poll updates SQLite, category preferences, replay buffer, and DDQN weights |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Database | SQLite (via `database.py`) |
| RL Agent | PyTorch — Dueling Double DQN |
| Similarity | scikit-learn TF-IDF + cosine similarity |
| Attention | OpenCV + MediaPipe Face Mesh |
| Auth | JWT tokens + SHA-256 password hashing |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Data | Pandas (CSV → SQLite import) |

---

## Folder Structure

```
news_recommender/
│
├── final/                              ← Main project root
│   │
│   ├── backend/
│   │   └── main.py                     ← FastAPI app, all API routes
│   │
│   ├── frontend/
│   │   ├── index.html                  ← Main feed UI
│   │   ├── login.html                  ← Login page
│   │   ├── signup.html                 ← Signup page
│   │   ├── app.js                      ← Feed logic, webcam, feedback, polls
│   │   ├── auth.js                     ← Login / signup JS
│   │   ├── styles.css                  ← Main stylesheet
│   │   └── auth.css                    ← Auth page styles
│   │
│   ├── data/
│   │   ├── news_transformed.csv        ← Place your news dataset here
│   │   └── user_profile_scored.csv     ← Place your user dataset here
│   │       (news_recommender.sqlite auto-generated on first run)
│   │
│   ├── models/
│   │   └── .gitkeep                    ← DDQN weights saved here per user
│   │       (ddqn_<user_id>.pt files created automatically)
│   │
│   ├── tests/
│   │   ├── simulate_session.py         ← Automated session simulation
│   │   └── deep_audit.py               ← Deep audit / integration tests
│   │
│   ├── auth.py                         ← JWT auth: signup, login, token verify
│   ├── database.py                     ← SQLite schema, CSV importer, helpers
│   ├── ddqn_agent.py                   ← Dueling Double DQN implementation
│   ├── recommender.py                  ← Core recommendation engine
│   ├── train_ddqn.py                   ← Offline DDQN training script
│   ├── utils.py                        ← Scoring formulas, clamp, similarity helpers
│   ├── webcam_attention.py             ← OpenCV + MediaPipe attention pipeline
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## Prerequisites

| Requirement | Minimum Version | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| Webcam *(optional)* | any USB / built-in | — |
| RAM | 4 GB+ | — |
| Disk space | 2 GB free (PyTorch) | — |

> **Windows users:** Use **Git Bash**, **CMD**, or **PowerShell** — all commands below work in any of them.

---

## Local Setup — Step by Step

### Step 1 — Clone the repository

```bash
git clone https://github.com/Dracula-5/news_recommender.git
cd news_recommender
```

---

### Step 2 — Create and activate a virtual environment

```bash
# Create the venv
python -m venv venv

# Windows CMD
venv\Scripts\activate.bat

# Windows PowerShell
venv\Scripts\Activate.ps1

# Git Bash / macOS / Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt after activation.

---

### Step 3 — Install dependencies

```bash
pip install -r final/requirements.txt
```

> This installs FastAPI, Uvicorn, PyTorch, OpenCV, MediaPipe, scikit-learn, pandas, and everything else.
> First install takes **3–5 minutes** because PyTorch is large.

---

### Step 4 — Add your dataset files

Place both CSV files into the `final/data/` folder:

```
final/data/news_transformed.csv
final/data/user_profile_scored.csv
```

The importer also checks these fallback locations:

```
final/news_transformed.csv
final/interaction_profile_scored_final.csv
```

The SQLite database is created automatically when the server starts for the first time.

---

### Step 5 — Start the backend server

```bash
cd final
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:

```
🚀 Starting backend...
⚠️  Webcam attention disabled. Set ENABLE_WEBCAM_ATTENTION=1 to enable.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

### Step 6 — Import your data (first run only)

Open your browser and go to:

```
http://localhost:8000/docs
```

Find **`POST /import_data`** → click **Try it out → Execute**.

Or run it from the terminal:

```bash
curl -X POST http://localhost:8000/import_data
```

This reads the CSVs and populates the SQLite database. **Only needed once.**

---

### Step 7 — Open the app

```
http://localhost:8000/app
```

1. Click **Sign up** to create an account
2. Fill in the onboarding form (interests, mood, time available, location)
3. Click **Launch feed →**
4. Read articles — like, dislike, or skip
5. Watch the feed adapt to your preferences in real time

---

### Step 8 — Enable webcam attention tracking (optional)

**Option A — Enable on server startup:**

```bash
# Windows CMD
set ENABLE_WEBCAM_ATTENTION=1
uvicorn backend.main:app --reload --port 8000

# Windows PowerShell
$env:ENABLE_WEBCAM_ATTENTION="1"
uvicorn backend.main:app --reload --port 8000

# Git Bash / macOS / Linux
ENABLE_WEBCAM_ATTENTION=1 uvicorn backend.main:app --reload --port 8000
```

**Option B — Enable from the UI:**

Click **Start webcam** in the top-right of the feed page. The camera panel shows a live annotated video stream and real-time metrics: brightness, eye openness, EAR, gaze score, head movement, and energy.

---

## How the Feed Works

```
Onboard → pick interests, mood, time available
    │
    ▼
Backend scores candidates:
  DDQN Q-value       (30%) + TF-IDF similarity (24%)
  Trending score     (14%) + Category preference (12%)
  Attention boost    (up to 25%) + Recency boost (up to 20%)
    │
    ▼
Articles split into 3 pools (reshuffled every 3 articles):
  ├── Liked category pool   → 45% of feed
  ├── Trending pool         → 25% of feed
  └── Random / explore      → 30% of feed
    │
    ▼
You interact:
  Like    → next 6 articles include 2–3 from that category
  Dislike → that category capped at 1 in next 5 articles
    │
    ▼
After 20 s of reading → poll overlay appears on the article card:
  "Do you like this article?"
  Yes (+0.10 preference) | No (−0.12 preference + skip)
    │
    ▼
DDQN trains on every interaction
→ Q-values update → better recommendations next time
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/app` | Serve frontend UI |
| `GET` | `/docs` | Swagger interactive docs |
| `POST` | `/auth/signup` | Create a new account |
| `POST` | `/auth/login` | Login, returns JWT token |
| `POST` | `/onboard` | Set user interests and context |
| `POST` | `/recommend` | Get top-k article recommendations |
| `POST` | `/feedback` | Submit like / dislike / skip with attention data |
| `POST` | `/poll_feedback` | Submit poll yes / no answer |
| `POST` | `/update_model` | Trigger manual DDQN training step |
| `GET` | `/user/{user_id}` | Get user profile and scores |
| `GET` | `/users` | List all known user IDs |
| `GET` | `/categories` | List all news categories |
| `POST` | `/import_data` | Import CSVs into SQLite |
| `GET` | `/attention` | Latest attention snapshot (JSON) |
| `GET` | `/attention/frame` | Single JPEG frame from webcam |
| `GET` | `/attention/stream` | MJPEG live camera stream |
| `POST` | `/attention/start` | Start webcam backend |
| `POST` | `/attention/stop` | Stop webcam backend |

Full interactive docs: `http://localhost:8000/docs`

---

## Database Tables

| Table | Purpose |
|---|---|
| `users` | Profiles, attention / interaction / exploration scores |
| `news` | Full article catalog from CSV |
| `interactions` | Every like / skip / poll with all attention + scoring signals |
| `user_category_prefs` | Per-user per-category preference (updated after every interaction) |
| `category_stats` | Impressions, clicks, likes, avg dwell, trending score per category |
| `user_memory` | Last 100 seen articles per user — prevents repetition |
| `agent_state` | DDQN step counter, epsilon, category order per user |
| `attention_events` | Raw webcam metrics snapshot per article view |

---

## Interaction Score Formula

```
interaction_score = 0.55 × click_val
                  + 0.10 × poll_val
                  + 0.20 × log_dwell
                  + 0.15 × long_term_history
```

| Component | Range | Meaning |
|---|---|---|
| `click_val` | 0 or 1 | Like = 1, skip / dislike = 0 |
| `poll_val` | 0 / 0.5 / 1 | Poll No / No answer / Poll Yes |
| `log_dwell` | 0 → 1 | Reading time normalized (60 s = 1.0) |
| `long_term_history` | 0 → 1 | User's historical preference for this category |

---

## Troubleshooting

**`ModuleNotFoundError` on startup**
```bash
# Make sure venv is active, then:
pip install -r final/requirements.txt
```

**No articles showing after onboarding**
```bash
curl -X POST http://localhost:8000/import_data
# Verify CSV files exist in final/data/ or final/
```

**`FileNotFoundError` for CSV datasets**
Place `news_transformed.csv` and `user_profile_scored.csv` (or `interaction_profile_scored_final.csv`) in `final/data/` or directly in `final/`.

**Camera won't start**
Close all apps using the camera (browsers, Zoom, Teams). If it still fails, edit `webcam_attention.py` and change `camera_index=0` to `camera_index=1`.

**Cannot connect to `localhost:8000`**
Make sure uvicorn is running and you're opening `http://localhost:8000/app` in the browser — not opening the HTML file directly.

**Poll not appearing**
The poll triggers after **20 seconds** on the same article. Watch the timer in the top bar.

**Login keeps redirecting to login page**
Token may have expired. Open browser DevTools → Application → Local Storage → Clear all, then log in again.

**PyTorch `FutureWarning` about `weights_only`**
Already handled in `ddqn_agent.py`. Safe to ignore if it still appears.

---

## Running Tests

```bash
cd final

# Simulate a full user session (headless, no browser needed)
python tests/simulate_session.py

# Deep system audit — checks all major components
python tests/deep_audit.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENABLE_WEBCAM_ATTENTION` | `0` | Set to `1` to auto-start webcam on server launch |

---


---

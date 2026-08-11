# Hyper-Personalized News Recommendation System

Production-style local project for a Human-In-The-Loop news recommender using a Dueling Double Deep Q Network, SQLite, FastAPI, and a browser UI.

## What It Does

- Converts the CSV datasets into SQLite tables.
- Recommends top-k news articles, with `k` constrained to 8-10.
- Chooses categories with a Dueling Double DQN and then ranks articles inside those categories.
- Updates SQLite, category preferences, replay memory, and model weights after every feedback event.
- Uses a weighted score from RL Q-value, TF-IDF similarity, trending clicks/likes, recent memory, diversity, and randomness.
- Blocks a category temporarily after two dislikes.
- Avoids repeating recently shown articles and caps repeated categories at 3 per batch.
- Provides FastAPI endpoints and a clean one-card-at-a-time UI.
- Includes optional OpenCV + MediaPipe attention tracking.

## Project Layout

```text
.
├── backend/
│   └── main.py
├── data/
│   └── news_recommender.sqlite        # created after import
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── models/
│   └── ddqn_<user>.pt                 # created after feedback/training
├── tests/
│   └── simulate_session.py
├── database.py
├── ddqn_agent.py
├── recommender.py
├── train_ddqn.py
├── utils.py
├── webcam_attention.py
└── requirements.txt
```

## Dataset

The importer accepts either location:

- `data/user_profile_scored.csv` or `interaction_profile_scored_final.csv`
- `data/news_transformed.csv`
- project-root copies of the same files

Your current workspace already has:

- `interaction_profile_scored_final.csv`
- `news_transformed.csv`

So the app can import them directly.

## Setup

Install Python 3.10+ if it is not already on PATH, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your machine uses the Windows launcher:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
```

## Import CSVs To SQLite

```powershell
python database.py
```

This creates `data/news_recommender.sqlite` with:

- `users`
- `news`
- `interactions`
- `user_category_prefs`
- `category_stats`
- `user_memory`
- `agent_state`

Use a clean rebuild when needed:

```powershell
python -c "from database import import_csvs; import_csvs(force=True)"
```

## Run The Backend And UI

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/app
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `POST /onboard`
- `POST /recommend`
- `POST /feedback`
- `POST /update_model`
- `POST /import_data`
- `GET /categories`
- `GET /attention`

Example recommendation request:

```json
{
  "user_id": "demo_user",
  "k": 8
}
```

Example feedback request:

```json
{
  "user_id": "demo_user",
  "news_id": "N55528",
  "time_spent": 9.5,
  "liked": 1,
  "scroll_depth": 0.9,
  "click_val": 1,
  "attention_score": 0.72,
  "final_score": 0.81,
  "similarity": 0.33,
  "trending": 0.15
}
```

## Warm-Start Training

```powershell
python train_ddqn.py --user-id demo_user --episodes 40
```

The DDQN architecture is:

- `Linear(5 -> 128) + ReLU`
- value stream: `Linear(128 -> 64) -> ReLU -> Linear(64 -> 1)`
- advantage stream: `Linear(128 -> 64) -> ReLU -> Linear(64 -> N_categories)`
- `Q(s,a) = V(s) + (A(s,a) - mean(A))`

Training uses replay buffer size 500, batch size 16, gamma 0.9, Adam, MSE loss, Double DQN target selection, and target network sync every 20 steps. Feedback is stored immediately; gradient updates begin once the replay buffer has at least 16 transitions.

## Simulate A Session

```powershell
python tests\simulate_session.py
```

This verifies:

- recommendations return at least 8 items
- no duplicate article IDs in a batch
- category repetition is capped at 3
- feedback updates model/database over time

## Optional Webcam Attention

Run attention in a separate terminal:

```powershell
python webcam_attention.py
```

The backend reads the latest in-process snapshot via `/attention`. For a production deployment, run the webcam worker inside the same app process or publish its snapshot over Redis/WebSocket. The scoring functions are already import-safe:

- brightness
- eye openness with EAR
- optical-flow movement
- energy score
- gaze score
- attention score

## Notes

- The app runs on CPU only.
- Candidate sampling is limited to small batches before ranking.
- SQLite is updated after every article feedback.
- Model weights are saved under `models/`.
- Trending uses global category clicks/likes with `log(1 + clicks)` smoothing.

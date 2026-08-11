# NeuroFeed

See the [repository README](../README.md) for the full write-up
(architecture, RecSys techniques, evaluation results, quick start).

Quick reference for this directory:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/seed_demo_data.py --import
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Or: `docker compose up --build`.

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point every module that reads config.get_settings() at isolated,
# throwaway SQLite files *before* anything imports config/database/auth/main
# — otherwise tests would run against (and mutate) the real dev databases.
# Both DBs need this, not just the main one: auth.py used to hardcode its
# own path with no override at all, so any test touching signup/login was
# silently writing real accounts into the dev auth.sqlite.
_TEST_DB_PATH = ROOT / "data" / f"test_{uuid.uuid4().hex[:8]}.sqlite"
_TEST_AUTH_DB_PATH = ROOT / "data" / f"test_auth_{uuid.uuid4().hex[:8]}.sqlite"
os.environ["NEUROFEED_DB_PATH"] = str(_TEST_DB_PATH)
os.environ["NEUROFEED_AUTH_DB_PATH"] = str(_TEST_AUTH_DB_PATH)
os.environ["NEUROFEED_ENV"] = "test"
os.environ["NEUROFEED_ENABLE_WEBCAM_ATTENTION"] = "false"
os.environ["NEUROFEED_RATE_LIMIT_REQUESTS"] = "10000"  # tests shouldn't trip rate limiting
os.environ["NEUROFEED_AUTH_RATE_LIMIT_REQUESTS"] = "10000"  # separate limiter on /auth/signup, /auth/login
os.environ["NEUROFEED_LOGIN_MAX_FAILURES"] = "10000"  # tests deliberately trigger failed logins

from database import import_csvs  # noqa: E402
from scripts.seed_demo_data import generate_news, generate_users, _rng  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _seeded_test_db():
    """Seed the isolated test DB once per test session with synthetic data."""
    rng = _rng(seed=123)
    news_df = generate_news(n_per_category=15, rng=rng)
    users_df = generate_users(n_users=10, rng=rng)

    data_dir = _TEST_DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    news_csv = data_dir / f"_test_news_{_TEST_DB_PATH.stem}.csv"
    users_csv = data_dir / f"_test_users_{_TEST_DB_PATH.stem}.csv"
    news_df.to_csv(news_csv, index=False)
    users_df.to_csv(users_csv, index=False)

    import database as _db

    # import_csvs() looks in fixed candidate locations — point it at our temp CSVs.
    _db.NEWS_CSV_CANDIDATES = [news_csv]
    _db.USER_CSV_CANDIDATES = [users_csv]

    import_csvs(_TEST_DB_PATH, force=True)

    yield

    # DDQNAgent checkpoint saves run on a shared background thread pool
    # (see ddqn_agent.py::_SAVE_EXECUTOR) and are fire-and-forget from the
    # caller's perspective — drain it before deleting the temp DB files
    # below, or a save still in flight can lose its race against unlink()
    # and log a spurious "no such table" from a thread with nothing left
    # to report it to.
    from ddqn_agent import _SAVE_EXECUTOR

    _SAVE_EXECUTOR.shutdown(wait=True)

    for path in (_TEST_DB_PATH, _TEST_AUTH_DB_PATH, news_csv, users_csv):
        path.unlink(missing_ok=True)
    for db in (_TEST_DB_PATH, _TEST_AUTH_DB_PATH):
        for suffix in ("-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)


@pytest.fixture
def db_path() -> Path:
    return _TEST_DB_PATH


@pytest.fixture
def engine(_seeded_test_db):
    from recommender import NewsRecommender

    return NewsRecommender(_TEST_DB_PATH)


@pytest.fixture
def api_client(_seeded_test_db):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def unique_user_id() -> str:
    return f"test_user_{uuid.uuid4().hex[:10]}"

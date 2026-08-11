from __future__ import annotations

import json
import sqlite3
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "news_recommender.sqlite"


USER_CSV_CANDIDATES = [
    DATA_DIR / "user_profile_scored.csv",
    DATA_DIR / "interaction_profile_scored_final.csv",
    ROOT / "user_profile_scored.csv",
    ROOT / "interaction_profile_scored_final.csv",
]
NEWS_CSV_CANDIDATES = [
    DATA_DIR / "news_transformed.csv",
    ROOT / "news_transformed.csv",
]


def find_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


@contextmanager
def db_connect(path: Path = DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache
    conn.execute("PRAGMA mmap_size=268435456") # 256 MB memory-mapped I/O
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    with db_connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_interest_text TEXT DEFAULT '',
                mood TEXT DEFAULT '',
                time_available REAL DEFAULT 10,
                time_of_day TEXT DEFAULT '',
                origin_location TEXT DEFAULT '',
                current_location TEXT DEFAULT '',
                attention_score REAL DEFAULT 0.5,
                interaction_score REAL DEFAULT 0.5,
                recent_activity REAL DEFAULT 0.5,
                exploration_signal REAL DEFAULT 0.2,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS news (
                news_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                location TEXT DEFAULT '',
                preferred_time TEXT DEFAULT '',
                age_group TEXT DEFAULT '',
                article_length REAL DEFAULT 0,
                abstract TEXT DEFAULT '',
                full_article TEXT DEFAULT '',
                url TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                news_id TEXT NOT NULL,
                category TEXT NOT NULL,
                time_spent REAL DEFAULT 0,
                liked INTEGER DEFAULT 0,
                skipped INTEGER DEFAULT 0,
                scroll_depth REAL DEFAULT 0,
                click_val REAL DEFAULT 0,
                reward REAL DEFAULT 0,
                attention_score REAL DEFAULT 0,
                interaction_score REAL DEFAULT 0,
                final_score REAL DEFAULT 0,
                q_value REAL DEFAULT 0,
                similarity REAL DEFAULT 0,
                trending REAL DEFAULT 0,
                brightness REAL DEFAULT 50,
                eye_openness REAL DEFAULT 70,
                movement REAL DEFAULT 20,
                energy_score REAL DEFAULT 63.5,
                gaze_score REAL DEFAULT 0.58,
                normalized_attention REAL DEFAULT 0.5,
                attention_source TEXT DEFAULT 'frontend',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS attention_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                news_id TEXT DEFAULT '',
                brightness REAL DEFAULT 50,
                eye_openness REAL DEFAULT 70,
                movement REAL DEFAULT 20,
                energy_score REAL DEFAULT 63.5,
                gaze_score REAL DEFAULT 0.58,
                attention_score REAL DEFAULT 0.62,
                normalized_attention REAL DEFAULT 0.62,
                face_detected INTEGER DEFAULT 0,
                source TEXT DEFAULT 'frontend',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_category_prefs (
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                preference REAL DEFAULT 0,
                dislikes INTEGER DEFAULT 0,
                blocked_until_step INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, category)
            );

            CREATE TABLE IF NOT EXISTS category_stats (
                category TEXT PRIMARY KEY,
                impressions INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                avg_time REAL DEFAULT 0,
                trending_score REAL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                news_id TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, news_id)
            );

            CREATE TABLE IF NOT EXISTS agent_state (
                user_id TEXT PRIMARY KEY,
                step INTEGER DEFAULT 0,
                epsilon REAL DEFAULT 0.15,
                category_order TEXT DEFAULT '[]',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Knowledge-graph: per-user category interest nodes (time-decayed)
            CREATE TABLE IF NOT EXISTS graph_edges (
                user_id      TEXT NOT NULL,
                category     TEXT NOT NULL,
                total_reward REAL DEFAULT 0,
                count        INTEGER DEFAULT 0,
                liked        INTEGER DEFAULT 0,
                last_time    REAL DEFAULT 0,
                PRIMARY KEY (user_id, category)
            );

            -- Knowledge-graph: co-interest edges between category pairs a
            -- user has liked (time-decayed independently of the nodes
            -- above — an association fades if it isn't reinforced, same
            -- as interest in a single category does).
            CREATE TABLE IF NOT EXISTS graph_co_interest_edges (
                user_id     TEXT NOT NULL,
                category_a  TEXT NOT NULL,
                category_b  TEXT NOT NULL,
                weight      REAL DEFAULT 0,
                last_time   REAL DEFAULT 0,
                PRIMARY KEY (user_id, category_a, category_b)
            );

            CREATE INDEX IF NOT EXISTS idx_news_category ON news(category);
            CREATE INDEX IF NOT EXISTS idx_news_location ON news(location);
            CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_memory_user_pos ON user_memory(user_id, position);
            CREATE INDEX IF NOT EXISTS idx_attention_user ON attention_events(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_cat_prefs_user_cat ON user_category_prefs(user_id, category);
            CREATE INDEX IF NOT EXISTS idx_cat_stats_category ON category_stats(category);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_user ON graph_edges(user_id);
            CREATE INDEX IF NOT EXISTS idx_co_interest_user ON graph_co_interest_edges(user_id);
            """
        )
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(interactions)").fetchall()}
        extra_cols = {
            "brightness": "REAL DEFAULT 50",
            "eye_openness": "REAL DEFAULT 70",
            "movement": "REAL DEFAULT 20",
            "energy_score": "REAL DEFAULT 63.5",
            "gaze_score": "REAL DEFAULT 0.58",
            "normalized_attention": "REAL DEFAULT 0.62",
            "attention_source": "TEXT DEFAULT 'frontend'",
            "poll_val": "REAL DEFAULT 0.5",
            "long_term_history": "REAL DEFAULT 0.2",
        }
        for col, definition in extra_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE interactions ADD COLUMN {col} {definition}")

        existing_graph_cols = {row["name"] for row in conn.execute("PRAGMA table_info(graph_edges)").fetchall()}
        if "avg_reward" not in existing_graph_cols:
            # Fixed-weight EMA of reward, tracked alongside the decayed
            # *sum* in total_reward. Node scoring is driven by this, not
            # total_reward/count — dividing a decayed sum by an
            # ever-growing raw count made a user's score sink toward zero
            # over a long history even with steady, undiminished
            # engagement (the numerator decays, the denominator never
            # does). An EMA with a fixed per-interaction weight doesn't
            # have that problem, and time-decay-toward-neutral is applied
            # separately at read time based on elapsed time since the last
            # interaction (see graph_memory.py::_node_score_locked).
            conn.execute("ALTER TABLE graph_edges ADD COLUMN avg_reward REAL DEFAULT 0")

        existing_user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "mood_updated_at" not in existing_user_cols:
            # When `mood` was last set — not the same as `updated_at`,
            # which touches on every attention/interaction score update
            # too. recommend()/feedback() decay mood's influence on
            # scoring based on this timestamp (see mood_signals.py) — a
            # mood set days ago shouldn't still be steering the feed.
            conn.execute("ALTER TABLE users ADD COLUMN mood_updated_at TEXT")


def normalize_user_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "user_id": "",
        "user_interest_text": "",
        "attention_score": 0.5,
        "interaction_score": 0.5,
        "origin_location": "",
        "current_location": "",
        "time_spent": 0,
        "click_val": 0,
        "scroll_depth": 0,
        "like_val": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    numeric_cols = [
        "attention_score",
        "interaction_score",
        "brightness",
        "eye_openness",
        "movement",
        "energy_score",
        "gaze_score",
        "total_score",
        "time_spent",
        "click_val",
        "scroll_depth",
        "like_val",
    ]
    df = clean_numeric_columns(df, numeric_cols)
    for col in ["attention_score", "interaction_score", "click_val", "scroll_depth", "like_val"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    for col in ["brightness", "eye_openness", "movement", "energy_score"]:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)
    if "gaze_score" in df.columns:
        df["gaze_score"] = df["gaze_score"].clip(0, 1)
    text_cols = ["user_id", "user_interest_text", "origin_location", "current_location"]
    df = clean_text_columns(df, text_cols, "unknown")
    return df


def normalize_news_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "news_id": "",
        "category": "general",
        "subcategory": "",
        "location": "",
        "preferred_time": "",
        "age_group": "",
        "article_length": 0,
        "abstract": "",
        "full_article": "",
        "url": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df = clean_numeric_columns(df, ["article_length"])
    if "article_length" in df.columns:
        df["article_length"] = df["article_length"].clip(lower=0)
    df = clean_text_columns(
        df,
        ["news_id", "category", "subcategory", "location", "preferred_time", "age_group", "abstract", "full_article", "url"],
        "",
    )
    df["category"] = df["category"].replace("", "general").str.lower().str.strip()
    return df


def clean_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.replace([math.inf, -math.inf], pd.NA)
        series = series.interpolate(method="linear", limit_direction="both")
        median = series.median(skipna=True)
        if pd.isna(median):
            median = 0
        df[col] = series.fillna(median)
    return df


def clean_text_columns(df: pd.DataFrame, cols: list[str], default: str) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(default).astype(str).str.strip()
        df[col] = df[col].replace({"nan": default, "None": default, "": default})
    return df


def import_csvs(db_path: Path = DB_PATH, force: bool = False, chunksize: int = 25_000) -> dict:
    init_db(db_path)
    user_csv = find_existing(USER_CSV_CANDIDATES)
    news_csv = find_existing(NEWS_CSV_CANDIDATES)

    with db_connect(db_path) as conn:
        if force:
            for table in [
                "users",
                "news",
                "interactions",
                "user_category_prefs",
                "category_stats",
                "user_memory",
                "agent_state",
                "attention_events",
            ]:
                conn.execute(f"DELETE FROM {table}")

        existing_news = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        existing_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    needs_news = force or existing_news == 0
    needs_users = force or existing_users == 0
    if (needs_news or needs_users) and (user_csv is None or news_csv is None):
        raise FileNotFoundError(
            "Expected profile/news CSVs in project root or data/. "
            "Need user_profile_scored.csv (or interaction_profile_scored_final.csv) and news_transformed.csv."
        )

    if needs_news:
        for chunk in pd.read_csv(news_csv, chunksize=chunksize):
            chunk = normalize_news_columns(chunk)
            rows = chunk[
                [
                    "news_id",
                    "category",
                    "subcategory",
                    "location",
                    "preferred_time",
                    "age_group",
                    "article_length",
                    "abstract",
                    "full_article",
                    "url",
                ]
            ].fillna("")
            with db_connect(db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO news
                    (news_id, category, subcategory, location, preferred_time, age_group,
                     article_length, abstract, full_article, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows.itertuples(index=False, name=None),
                )

    if needs_users:
        usecols = None
        for chunk in pd.read_csv(user_csv, chunksize=chunksize, usecols=usecols):
            chunk = normalize_user_columns(chunk)
            user_rows = chunk[
                [
                    "user_id",
                    "user_interest_text",
                    "attention_score",
                    "interaction_score",
                    "origin_location",
                    "current_location",
                ]
            ].drop_duplicates("user_id").fillna("")
            with db_connect(db_path) as conn:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO users
                    (user_id, user_interest_text, attention_score, interaction_score,
                     origin_location, current_location)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    user_rows.itertuples(index=False, name=None),
                )

                feedback_rows = chunk[
                    [
                        "user_id",
                        "time_spent",
                        "click_val",
                        "scroll_depth",
                        "like_val",
                        "attention_score",
                        "interaction_score",
                    ]
                ].fillna(0)
                for row in feedback_rows.itertuples(index=False):
                    # Historical profile rows do not have news_id/category, so they seed only user activity.
                    conn.execute(
                        """
                        UPDATE users
                        SET recent_activity = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                        """,
                        (float(row.interaction_score), str(row.user_id)),
                    )

    with db_connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO category_stats(category)
            SELECT DISTINCT category FROM news
            """
        )
        categories = [r[0] for r in conn.execute("SELECT category FROM category_stats")]
        user_ids = [r[0] for r in conn.execute("SELECT user_id FROM users LIMIT 200")]
        for user_id in user_ids:
            conn.execute(
                "INSERT OR IGNORE INTO agent_state(user_id, category_order) VALUES (?, ?)",
                (user_id, json.dumps(categories)),
            )
    return {
        "db_path": str(db_path),
        "user_csv": str(user_csv),
        "news_csv": str(news_csv),
    }


def fetch_categories(db_path: Path = DB_PATH) -> list[str]:
    init_db(db_path)
    with db_connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT category FROM news ORDER BY category").fetchall()
    return [row[0] for row in rows]


if __name__ == "__main__":
    print(import_csvs(force=False))

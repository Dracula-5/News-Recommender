"""
Synthetic demo-data generator.

NeuroFeed's importer (``database.import_csvs``) expects a MIND-style dataset
(``news_transformed.csv`` + ``user_profile_scored.csv``) that isn't part of
this repo — it's proprietary/large and gitignored, same as any real
production dataset would be. This script generates a small, fully synthetic
stand-in with the same shape so the project is self-contained: clone it,
seed it, run it, no external download required.

The generated articles are template-based (not scraped, not real headlines)
but topically coherent per category, which matters — the NLP retrieval
engine (BM25/TF-IDF/LSA) and the evaluation harness both need real lexical
signal to produce meaningful, non-trivial results.

Usage:
    python scripts/seed_demo_data.py                 # write CSVs only
    python scripts/seed_demo_data.py --import         # also import into SQLite
    python scripts/seed_demo_data.py --import --force # wipe + reimport
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

# ── Category vocabulary banks ──────────────────────────────────────────
# Each category gets its own subject/action/object pools so generated
# headlines cluster lexically the way a real news corpus would.

CATEGORY_BANKS: dict[str, dict[str, list[str]]] = {
    "technology": {
        "subcategory": ["ai", "hardware", "startups", "cybersecurity", "mobile"],
        "subject": ["a Bay Area startup", "the chipmaker", "a cloud provider", "an AI research lab", "a security firm"],
        "action": ["unveils", "patches a flaw in", "raises funding for", "open-sources", "benchmarks"],
        "object": ["a faster inference chip", "a large language model", "a zero-day exploit", "a developer toolkit", "an autonomous agent framework"],
    },
    "business": {
        "subcategory": ["markets", "startups", "economy", "earnings", "trade"],
        "subject": ["the central bank", "a retail giant", "the finance ministry", "a logistics firm", "an index fund"],
        "action": ["raises rates amid", "reports earnings above", "cuts jobs after", "expands into", "warns of"],
        "object": ["persistent inflation", "analyst forecasts", "a supply-chain slowdown", "a new export market", "a currency slump"],
    },
    "sports": {
        "subcategory": ["cricket", "football", "tennis", "olympics", "athletics"],
        "subject": ["the national team", "a veteran striker", "the reigning champion", "a rookie sprinter", "the home side"],
        "action": ["clinches victory in", "is ruled out of", "sets a new record at", "stages a comeback in", "signs a new contract before"],
        "object": ["the season opener", "the quarter-final", "the regional championship", "a marquee derby", "the qualifying round"],
    },
    "health": {
        "subcategory": ["public-health", "nutrition", "mental-health", "fitness", "medicine"],
        "subject": ["researchers", "a clinical trial", "the health ministry", "a new study", "public-health officials"],
        "action": ["link", "flag concerns over", "recommend limiting", "report progress on", "issue guidance about"],
        "object": ["sleep and screen time", "a seasonal virus strain", "processed sugar intake", "a new vaccine candidate", "workplace stress"],
    },
    "science": {
        "subcategory": ["space", "climate", "biology", "physics", "environment"],
        "subject": ["astronomers", "a climate model", "a research team", "a space agency", "marine biologists"],
        "action": ["detect", "confirm", "map", "measure a rise in", "publish findings on"],
        "object": ["a distant exoplanet", "coral reef recovery", "glacial melt rates", "a gravitational anomaly", "deep-sea biodiversity"],
    },
    "entertainment": {
        "subcategory": ["movies", "music", "television", "celebrity", "streaming"],
        "subject": ["a streaming platform", "the director", "a chart-topping artist", "the studio", "a veteran actor"],
        "action": ["greenlights", "teases", "postpones the release of", "wraps filming on", "announces a sequel to"],
        "object": ["a limited series", "a world tour", "an award-season contender", "a long-awaited biopic", "a fan-favorite franchise"],
    },
    "politics": {
        "subcategory": ["elections", "policy", "diplomacy", "legislation", "local-government"],
        "subject": ["the coalition", "a state legislature", "the opposition", "a policy panel", "city officials"],
        "action": ["debates", "passes a bill on", "delays a vote on", "reaches consensus on", "faces scrutiny over"],
        "object": ["infrastructure spending", "a redistricting plan", "trade regulations", "a public-transit expansion", "campaign finance rules"],
    },
    "world": {
        "subcategory": ["asia", "europe", "americas", "africa", "middle-east"],
        "subject": ["regional leaders", "a UN envoy", "border authorities", "a trade delegation", "humanitarian agencies"],
        "action": ["meet to discuss", "monitor", "reopen talks on", "coordinate relief after", "sign an accord on"],
        "object": ["cross-border trade", "a ceasefire framework", "flood relief efforts", "energy cooperation", "a refugee resettlement plan"],
    },
    "education": {
        "subcategory": ["higher-ed", "k12", "policy", "research", "online-learning"],
        "subject": ["a university system", "the education board", "researchers", "a school district", "an ed-tech platform"],
        "action": ["pilots", "expands access to", "reports gains from", "revises guidelines for", "invests in"],
        "object": ["a hybrid-learning program", "need-based scholarships", "standardized testing", "early-literacy initiatives", "campus mental-health services"],
    },
    "lifestyle": {
        "subcategory": ["travel", "food", "home", "personal-finance", "culture"],
        "subject": ["a travel survey", "local chefs", "a design collective", "personal-finance experts", "a cultural festival"],
        "action": ["highlights", "reinvent", "rank", "recommend", "celebrate"],
        "object": ["off-season destinations", "regional comfort food", "small-space living", "simple budgeting habits", "community art projects"],
    },
}

LOCATIONS = [
    "India", "United States", "United Kingdom", "Germany", "Japan",
    "Australia", "Brazil", "South Africa", "Canada", "Singapore",
]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
TIME_SLOTS = ["morning", "afternoon", "evening", "late_night"]
MOODS = ["curious", "focused", "relaxed", "busy", "energetic", "tired"]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate_news(n_per_category: int, rng: random.Random) -> pd.DataFrame:
    rows = []
    news_idx = 1
    for category, bank in CATEGORY_BANKS.items():
        for _ in range(n_per_category):
            subj = rng.choice(bank["subject"])
            act = rng.choice(bank["action"])
            obj = rng.choice(bank["object"])
            subcat = rng.choice(bank["subcategory"])
            headline = f"{subj.capitalize()} {act} {obj}"
            abstract = (
                f"{headline}. Officials and analysts covering {subcat.replace('-', ' ')} "
                f"say the development could shape the {category} outlook in the coming weeks."
            )
            full_article = (
                f"{abstract} Early reaction from the {subcat.replace('-', ' ')} community has been mixed, "
                f"with some pointing to precedent from similar {category} stories this year. "
                f"Coverage will be updated as the situation develops."
            )
            rows.append(
                {
                    "news_id": f"N{news_idx:05d}",
                    "category": category,
                    "subcategory": subcat,
                    "location": rng.choice(LOCATIONS),
                    "preferred_time": rng.choice(TIME_SLOTS),
                    "age_group": rng.choice(AGE_GROUPS),
                    "article_length": len(full_article.split()),
                    "abstract": abstract,
                    "full_article": full_article,
                    "url": "",
                }
            )
            news_idx += 1
    return pd.DataFrame(rows)


def generate_users(n_users: int, rng: random.Random) -> pd.DataFrame:
    categories = list(CATEGORY_BANKS.keys())
    rows = []
    for i in range(n_users):
        n_interests = rng.randint(1, 3)
        interests = rng.sample(categories, n_interests)
        rows.append(
            {
                "user_id": f"demo_user_{i + 1:03d}",
                "user_interest_text": " ".join(interests),
                "attention_score": round(rng.uniform(0.35, 0.75), 3),
                "interaction_score": round(rng.uniform(0.3, 0.7), 3),
                "origin_location": rng.choice(LOCATIONS),
                "current_location": rng.choice(LOCATIONS),
                "mood": rng.choice(MOODS),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=40, help="Articles generated per category (default: 40)")
    parser.add_argument("--users", type=int, default=40, help="Synthetic users generated (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42, deterministic output)")
    parser.add_argument("--import", dest="do_import", action="store_true", help="Import the generated CSVs into SQLite after writing them")
    parser.add_argument("--force", action="store_true", help="Wipe existing DB tables before importing (only with --import)")
    args = parser.parse_args()

    rng = _rng(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    news_df = generate_news(args.per_category, rng)
    users_df = generate_users(args.users, rng)

    news_path = DATA_DIR / "news_transformed.csv"
    users_path = DATA_DIR / "interaction_profile_scored_final.csv"
    news_df.to_csv(news_path, index=False)
    users_df.to_csv(users_path, index=False)

    print(f"Wrote {len(news_df)} articles -> {news_path}")
    print(f"Wrote {len(users_df)} users    -> {users_path}")

    if args.do_import:
        from database import DB_PATH, import_csvs

        result = import_csvs(DB_PATH, force=args.force)
        print(f"Imported into {DB_PATH}: {result}")


if __name__ == "__main__":
    main()

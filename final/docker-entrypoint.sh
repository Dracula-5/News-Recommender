#!/bin/sh
set -eu

# Runs as root (see Dockerfile). Docker creates named volumes fresh at
# mount time, independent of whatever the image had baked in — observed
# here as both wrong ownership (root, not appuser) *and* a missing write
# bit (mode 555) on the fresh volume, so both have to be fixed at
# container start, not at build time, or the unprivileged app user can't
# write to /app/data at all.
chown -R appuser:appuser /app/data /app/models
chmod -R u+rwX /app/data /app/models

# First boot (fresh volume, empty DB): seed the synthetic demo dataset so
# the container is useful the moment it comes up, no manual data-loading
# step required. Skipped entirely if the DB already has articles — either
# a previous run already seeded it, or a real dataset was mounted in and
# imported; either way this must never overwrite it.
NEWS_COUNT=$(gosu appuser python -c "
from database import DB_PATH, db_connect, init_db
init_db(DB_PATH)
with db_connect(DB_PATH) as conn:
    print(conn.execute('SELECT COUNT(*) FROM news').fetchone()[0])
" 2>/dev/null || echo 0)

if [ "$NEWS_COUNT" = "0" ]; then
    echo "No articles in DB — seeding synthetic demo dataset..."
    gosu appuser python scripts/seed_demo_data.py --import
fi

exec gosu appuser "$@"

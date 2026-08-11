FROM python:3.11-slim AS base

# OpenCV/MediaPipe need a handful of shared libs that "slim" doesn't ship.
# No camera device is passed into the container, so webcam attention stays
# off by default (NEUROFEED_ENABLE_WEBCAM_ATTENTION=false) — the app falls
# back to its no-face-detected state, same as any browser without camera
# permission. gosu lets the entrypoint start as root just long enough to
# fix volume ownership, then drop to an unprivileged user for the app.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so `docker build` caches this layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app \
    && chmod +x /app/docker-entrypoint.sh

ENV NEUROFEED_DB_PATH=/app/data/news_recommender.sqlite \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Stays root here on purpose — docker-entrypoint.sh fixes ownership on the
# mounted volumes (which Docker creates as root, independent of anything
# baked into the image) and then execs the app as `appuser` via gosu.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

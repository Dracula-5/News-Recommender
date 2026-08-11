"""
Structured logging setup, shared by the API process and standalone scripts.

Two things this fixes vs. the old ad hoc ``print()`` calls scattered around
the codebase:

1. Plain ``print()`` with emoji crashes on a default Windows console (cp1252
   can't encode most emoji) — that's a real, reproducible startup crash, not
   a style nit. Routing everything through ``logging`` with a plain-ASCII
   formatter sidesteps it entirely.
2. Every log line now carries a logger name, level, and timestamp, and can
   be redirected/aggregated the way you'd actually run this in production
   (stdout, picked up by Docker/whatever log driver sits in front of it).
"""
from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet down noisy third-party loggers unless we're at DEBUG.
    if level.upper() != "DEBUG":
        for noisy in ("uvicorn.access", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

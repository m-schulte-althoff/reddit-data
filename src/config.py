"""Shared configuration for the reddit-data pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Time window ──────────────────────────────────────────────────────────────
# 18 months before 2022-11-30 and 12 months after, interpreted literally.
# Adjust these two values to change the extraction window.
START_TS = datetime(2022, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
END_EXCLUSIVE_TS = datetime(2022, 12, 30, 0, 0, 0, tzinfo=timezone.utc)

START_EPOCH: int = int(START_TS.timestamp())
END_EXCLUSIVE_EPOCH: int = int(END_EXCLUSIVE_TS.timestamp())

# Derived month range (inclusive on both ends).
START_MONTH: tuple[int, int] = (START_TS.year, START_TS.month)
_end_incl = END_EXCLUSIVE_TS - timedelta(seconds=1)
END_MONTH: tuple[int, int] = (_end_incl.year, _end_incl.month)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path.cwd()
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# ── Arctic Shift torrent ────────────────────────────────────────────────────
TORRENT_INFOHASH = "9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4"
TRACKERS: list[str] = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://tracker.openbittorrent.com:6969/announce",
    "udp://tracker.torrent.eu.org:451/announce",
    "udp://open.stealth.si:80/announce",
]

# ── Hugging Face ─────────────────────────────────────────────────────────────
HF_REPO_ID = "open-index/arctic"
HF_REPO_TYPE = "dataset"

# ── Processing parameters ───────────────────────────────────────────────────
DELETE_RAW_AFTER_FILTER: bool = False
METADATA_POLL_SECONDS: int = 3
DOWNLOAD_POLL_SECONDS: int = 10
ZSTD_MAX_WINDOW_SIZE: int = 2**31
OUTPUT_ZSTD_LEVEL: int = 9

# ── Analysis ─────────────────────────────────────────────────────────────────
SAMPLE_SIZE: int = 500
RANDOM_SEED: int = 42

"""Shared configuration for the reddit-data pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Time window ──────────────────────────────────────────────────────────────
# 18 months before 2022-11-30 and 12 months after, interpreted literally.
# Adjust these two values to change the extraction window.
START_TS = datetime(2022, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
END_EXCLUSIVE_TS = datetime(2022, 4, 1, 0, 0, 0, tzinfo=timezone.utc)

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
LEGACY_BUNDLE_END_MONTH: tuple[int, int] = (2023, 12)
MONTHLY_TORRENT_INFOHASHES: dict[str, str] = {
    "2024-01": "ac88546145ca3227e2b90e51ab477c4527dd8b90",
    "2024-02": "5969ae3e21bb481fea63bf649ec933c222c1f824",
    "2024-03": "deef710de36929e0aa77200fddda73c86142372c",
    "2024-04": "ad4617a3e9c1f52405197fc088b28a8018e12a7a",
    "2024-05": "4f60634d96d35158842cd58b495dc3b444d78b0d",
    "2024-06": "dcdecc93ca9a9d758c045345112771cef5b4989a",
    "2024-07": "6e5300446bd9b328d0b812cdb3022891e086d9ec",
    "2024-08": "8c2d4b00ce8ff9d45e335bed106fe9046c60adb0",
    "2024-09": "43a6e113d6ecacf38e58ecc6caa28d68892dd8af",
    "2024-10": "507dfcda29de9936dd77ed4f34c6442dc675c98f",
    "2024-11": "a1b490117808d9541ab9e3e67a3447e2f4f48f01",
    "2024-12": "eb2017da9f63a49460dde21a4ebe3b7c517f3ad9",
    "2025-01": "4fd14d4c3d792e0b1c5cf6b1d9516c48ba6c4a24",
    "2025-02": "2f873e0b15da5ee29b63e586c0ab1dedd3508870",
    "2025-03": "69d5e046e15c02182430879f50d62b18fe1404fb",
    "2025-04": "552f34df5b830d18f98b69541e7e84f2658346b9",
    "2025-05": "186a0f85a52ff4f1b08677cd312423ace9b34976",
    "2025-06": "bec5590bd3bc6c0f2d868f36ec92bec1aff4480e",
    "2025-07": "b6a7ccf72368a7d39c018c423e01bc15aa551122",
    "2025-08": "c71a97c1f7f676c56963c4e15a81f20afb0109be",
    "2025-09": "a92ce24b4180e4aa9295353f4d26f050031e3058",
    "2025-10": "cb4fa22ea76ea0a2bb38885b27323c94a5d9d16c",
    "2025-11": "2d056b22743718ac81915f25b094b6226668663f",
    "2025-12": "481bf2eac43172ae724fd6c75dbcb8e27de77734",
    "2026-01": "8412b89151101d88c915334c45d9c223169a1a60",
    "2026-02": "c5ba00048236b60f819dbf010e9034d24fc291fb",
}
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

# ── Filtering ────────────────────────────────────────────────────────────────
INPUT_DIR = BASE_DIR / "input"
SUBREDDIT_LIST_PATH = INPUT_DIR / "subreddit-list-Chan-2025.txt"
FILTER_READ_BUFFER: int = 16 * 1024 * 1024  # 16 MB read buffer

# ── Analysis ─────────────────────────────────────────────────────────────────
SAMPLE_SIZE: int = 500
RANDOM_SEED: int = 42

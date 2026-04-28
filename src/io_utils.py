"""Shared I/O helpers for streaming filtered Reddit exports."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import orjson
import zstandard as zstd

from src.config import PROCESSED_DIR, ZSTD_MAX_WINDOW_SIZE


def epoch_to_month(epoch: int) -> str:
    """Convert a UTC epoch timestamp to ``YYYY-MM``."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def extract_created_utc(record: Mapping[str, Any]) -> int | None:
    """Return ``created_utc`` as an integer epoch when possible."""
    value = record.get("created_utc")
    if value is None:
        return None
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None


def stream_zst(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a .zst JSONL file."""
    dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
    with path.open("rb") as fin:
        with dctx.stream_reader(fin) as zin:
            buf = io.BufferedReader(zin)
            for line in buf:
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue


def discover_filtered_paths(kind: str, *, processed_dir: Path | None = None) -> list[Path]:
    """Find filtered ``.jsonl.zst`` files for ``kind`` in ``processed_dir``."""
    base_dir = processed_dir or PROCESSED_DIR
    paths = sorted(base_dir.glob(f"filter-{kind}-*.jsonl.zst"))
    return [path for path in paths if not path.name.endswith(".progress.json")]


def fingerprint_paths(paths: Sequence[Path]) -> dict[str, dict[str, int | str]]:
    """Build a stable fingerprint payload from file names, size, and mtime."""
    fingerprint: dict[str, dict[str, int | str]] = {}
    for path in sorted(paths):
        stat = path.stat()
        fingerprint[path.name] = {
            "size": stat.st_size,
            "mtime": str(stat.st_mtime),
        }
    return fingerprint


def fingerprint_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a fingerprint payload."""
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def month_to_parts(month: str) -> tuple[int, int]:
    """Split ``YYYY-MM`` into ``(year, month)``."""
    year, month_num = month.split("-", 1)
    return int(year), int(month_num)


def iter_month_range(start_month: str, end_month: str) -> list[str]:
    """Return all months from ``start_month`` to ``end_month`` inclusive."""
    year, month_num = month_to_parts(start_month)
    end_year, end_month_num = month_to_parts(end_month)

    months: list[str] = []
    while (year, month_num) <= (end_year, end_month_num):
        months.append(f"{year:04d}-{month_num:02d}")
        month_num += 1
        if month_num == 13:
            year += 1
            month_num = 1
    return months


def months_since(month: str, *, reference: str) -> int:
    """Return the integer month distance from ``reference`` to ``month``."""
    year, month_num = month_to_parts(month)
    ref_year, ref_month = month_to_parts(reference)
    return (year - ref_year) * 12 + (month_num - ref_month)


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning 0.0 when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def is_deleted_removed(value: str | None) -> bool:
    """Return whether a text field contains a deleted/removed placeholder."""
    if value is None:
        return False
    return value.strip().casefold() in {"[deleted]", "[removed]"}

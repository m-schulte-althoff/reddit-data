"""Descriptive overview of filtered (subreddit-specific) Reddit data.

Streams through compressed JSONL files in ``data/processed/`` to compute:
- Total post counts (aggregated and per subreddit).
- Monthly time series for trend analysis.
- Summary statistics (mean/median/std per subreddit).

Never loads all data into memory — uses single-pass streaming.
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import orjson
import zstandard as zstd

from src.config import (
    PROCESSED_DIR,
    ZSTD_MAX_WINDOW_SIZE,
)

log = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class DescribeResult:
    """Accumulator for descriptive statistics over filtered data."""

    kind: str
    total_records: int = 0
    parse_errors: int = 0
    subreddit_counts: Counter[str] = field(default_factory=Counter)
    monthly_counts: Counter[str] = field(default_factory=Counter)
    subreddit_monthly_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    min_ts: int | None = None
    max_ts: int | None = None

    def update_ts(self, ts: int) -> None:
        if self.min_ts is None or ts < self.min_ts:
            self.min_ts = ts
        if self.max_ts is None or ts > self.max_ts:
            self.max_ts = ts

    def time_range(self) -> tuple[str | None, str | None]:
        """Return ISO-formatted min/max timestamps."""
        def _fmt(e: int | None) -> str | None:
            return datetime.fromtimestamp(e, tz=timezone.utc).strftime("%Y-%m-%d") if e else None
        return _fmt(self.min_ts), _fmt(self.max_ts)

    def sorted_months(self) -> list[str]:
        """All months with data, sorted chronologically."""
        return sorted(self.monthly_counts.keys())

    def sorted_subreddits(self) -> list[str]:
        """Subreddits sorted alphabetically (case-insensitive)."""
        return sorted(self.subreddit_counts.keys(), key=str.lower)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _epoch_to_month(epoch: int) -> str:
    """Convert epoch seconds to ``YYYY-MM`` string."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def _extract_created_utc(record: dict) -> int | None:
    value = record.get("created_utc")
    if value is None:
        return None
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except Exception:
        return None


# ── Core streaming engine ───────────────────────────────────────────────────


def describe_filtered(
    kind: str,
    input_paths: list[Path] | None = None,
) -> DescribeResult:
    """Stream filtered .zst files and compute descriptive statistics.

    Parameters
    ----------
    kind
        ``"comments"`` or ``"submissions"`` — used as label.
    input_paths
        Explicit list of .zst files to process.  When ``None``, discovers
        all ``filter-{kind}-*.jsonl.zst`` files in ``PROCESSED_DIR``.
    """
    if input_paths is None:
        input_paths = sorted(PROCESSED_DIR.glob(f"filter-{kind}-*.jsonl.zst"))
        # Exclude progress sidecars.
        input_paths = [p for p in input_paths if not p.name.endswith(".progress.json")]

    if not input_paths:
        log.warning("No filtered files found for %s in %s", kind, PROCESSED_DIR)
        return DescribeResult(kind=kind)

    result = DescribeResult(kind=kind)

    for path in input_paths:
        log.info("Scanning %s …", path.name)
        dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
        with path.open("rb") as fin:
            with dctx.stream_reader(fin) as zin:
                buf = io.BufferedReader(zin)
                for line in buf:
                    if not line.strip():
                        continue
                    try:
                        obj = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        result.parse_errors += 1
                        continue

                    result.total_records += 1

                    ts = _extract_created_utc(obj)
                    if ts is not None:
                        result.update_ts(ts)
                        month = _epoch_to_month(ts)
                        result.monthly_counts[month] += 1
                    else:
                        month = "unknown"

                    sub = obj.get("subreddit", "unknown")
                    result.subreddit_counts[sub] += 1
                    result.subreddit_monthly_counts[(sub, month)] += 1

    log.info(
        "%s: %d records across %d subreddits, %d months",
        kind,
        result.total_records,
        len(result.subreddit_counts),
        len(result.monthly_counts),
    )
    return result

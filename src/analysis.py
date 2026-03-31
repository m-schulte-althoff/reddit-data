"""Aggregate descriptive statistics and sampling for raw Arctic Shift data.

Streams through compressed JSONL files in a single pass to compute stats
and optionally collect a reservoir sample — never loads all data into memory.
"""

from __future__ import annotations

import io
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import orjson
import zstandard as zstd

from src.config import (
    END_EXCLUSIVE_EPOCH,
    RAW_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    START_EPOCH,
    ZSTD_MAX_WINDOW_SIZE,
)

log = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    """Accumulator for single-pass descriptive statistics."""

    total_rows: int = 0
    rows_in_window: int = 0
    parse_errors: int = 0
    missing_created_utc: int = 0
    min_created_utc: int | None = None
    max_created_utc: int | None = None
    subreddit_counts: Counter[str] = field(default_factory=Counter)
    author_counts: Counter[str] = field(default_factory=Counter)
    score_sum: int = 0
    score_count: int = 0
    score_min: int | None = None
    score_max: int | None = None

    def update_ts(self, ts: int) -> None:
        if self.min_created_utc is None or ts < self.min_created_utc:
            self.min_created_utc = ts
        if self.max_created_utc is None or ts > self.max_created_utc:
            self.max_created_utc = ts

    def update_score(self, score: int) -> None:
        self.score_sum += score
        self.score_count += 1
        if self.score_min is None or score < self.score_min:
            self.score_min = score
        if self.score_max is None or score > self.score_max:
            self.score_max = score

    def as_dict(self) -> dict:
        """Serialisable summary."""
        return {
            "total_rows_scanned": self.total_rows,
            "rows_in_window": self.rows_in_window,
            "parse_errors": self.parse_errors,
            "missing_created_utc": self.missing_created_utc,
            "min_created_utc": _epoch_to_iso(self.min_created_utc),
            "max_created_utc": _epoch_to_iso(self.max_created_utc),
            "unique_subreddits": len(self.subreddit_counts),
            "top_subreddits_20": self.subreddit_counts.most_common(20),
            "unique_authors": len(self.author_counts),
            "top_authors_20": self.author_counts.most_common(20),
            "score_count": self.score_count,
            "score_sum": self.score_sum,
            "score_mean": (
                round(self.score_sum / self.score_count, 2)
                if self.score_count
                else None
            ),
            "score_min": self.score_min,
            "score_max": self.score_max,
        }


def _epoch_to_iso(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _extract_created_utc(record: dict) -> int | None:
    value = record.get("created_utc")
    if value is None:
        return None
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except Exception:
        return None


# ── Core streaming engine ───────────────────────────────────────────────────

def _resolve_paths(kind: str) -> list[Path]:
    """Return sorted raw .zst paths for the configured month range."""
    from src.arctic_shift import build_target_paths

    target_paths = build_target_paths()
    paths = [RAW_DIR / r for r in target_paths[kind]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Raw files missing: {[str(m) for m in missing]}  — run 'download' first."
        )
    return paths


def _stream_analyse(
    paths: list[Path],
    *,
    collect_sample: bool = False,
    sample_n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[StreamStats, list[dict]]:
    """Single-pass: compute stats and optionally reservoir-sample *sample_n* records."""
    stats = StreamStats()
    reservoir: list[dict] = []
    rng = random.Random(seed)
    idx = 0  # running count of in-window records

    for path in paths:
        log.info("Scanning %s …", path.name)
        dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
        with path.open("rb") as fin:
            with dctx.stream_reader(fin) as zin:
                buf = io.BufferedReader(zin)
                for line in buf:
                    if not line.strip():
                        continue
                    stats.total_rows += 1
                    try:
                        obj = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        stats.parse_errors += 1
                        continue
                    ts = _extract_created_utc(obj)
                    if ts is None:
                        stats.missing_created_utc += 1
                        continue
                    if not (START_EPOCH <= ts < END_EXCLUSIVE_EPOCH):
                        continue

                    # ── in-window record ──
                    stats.rows_in_window += 1
                    stats.update_ts(ts)

                    sub = obj.get("subreddit", "")
                    if sub:
                        stats.subreddit_counts[sub] += 1
                    author = obj.get("author", "")
                    if author:
                        stats.author_counts[author] += 1
                    score = obj.get("score")
                    if score is not None:
                        try:
                            stats.update_score(int(score))
                        except (ValueError, TypeError):
                            pass

                    # Reservoir sampling (Vitter's Algorithm R).
                    if collect_sample:
                        if idx < sample_n:
                            reservoir.append(obj)
                        else:
                            j = rng.randint(0, idx)
                            if j < sample_n:
                                reservoir[j] = obj
                        idx += 1

    return stats, reservoir


# ── Public API ───────────────────────────────────────────────────────────────

def analyse(kind: str) -> dict:
    """Return descriptive statistics for *kind* ('comments' or 'submissions')."""
    paths = _resolve_paths(kind)
    stats, _ = _stream_analyse(paths, collect_sample=False)
    summary = stats.as_dict()
    summary["kind"] = kind
    log.info(
        "%s: %d rows in window out of %d scanned",
        kind,
        stats.rows_in_window,
        stats.total_rows,
    )
    return summary


def sample(kind: str, n: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> list[dict]:
    """Return a reservoir sample of *n* in-window records."""
    paths = _resolve_paths(kind)
    _, reservoir = _stream_analyse(paths, collect_sample=True, sample_n=n, seed=seed)
    log.info("%s: sampled %d records", kind, len(reservoir))
    return reservoir


def sample_dataframe(
    kind: str,
    n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> object:
    """Return the reservoir sample as a ``pandas.DataFrame``."""
    import pandas as pd

    records = sample(kind, n=n, seed=seed)
    return pd.DataFrame(records)

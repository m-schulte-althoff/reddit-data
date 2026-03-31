"""Aggregate descriptive statistics and sampling for raw Arctic Shift data.

Streams through compressed JSONL files in a single pass to compute stats
and optionally collect a reservoir sample — never loads all data into memory.

Performance features:
- ``sample()`` / ``sample_dataframe()`` skip stat accumulation entirely.
- Lazy parsing: ``created_utc`` is extracted via byte-level regex; full JSON
  parsing is only done for the ~n records that enter the reservoir.
- Algorithm L (Li 1994): skip-ahead reservoir sampling avoids per-record RNG.
- When multiple .zst files are involved, sampling runs in parallel processes.
- ``months`` parameter lets callers restrict to a subset of files (faster).
- ``sample_dataframe()`` transparently caches to Parquet under ``output/tables/``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
    TABLES_DIR,
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

def _resolve_paths(
    kind: str,
    months: list[tuple[int, int]] | None = None,
) -> list[Path]:
    """Return sorted raw .zst paths for the configured month range.

    If *months* is given (e.g. ``[(2022, 10)]``), only those months are
    included — useful for faster exploratory sampling.
    """
    from src.arctic_shift import build_target_paths, iter_months, MonthRef

    if months is not None:
        from src.config import START_MONTH, END_MONTH

        all_months = iter_months(START_MONTH, END_MONTH)
        valid = {MonthRef(y, m) for y, m in months} & set(all_months)
        if not valid:
            raise ValueError(
                f"None of {months} fall within the configured range. "
                f"Available: {[(m.year, m.month) for m in all_months]}"
            )
        prefix = "comments" if kind == "comments" else "submissions"
        tag = "RC" if kind == "comments" else "RS"
        relpaths = [f"reddit/{prefix}/{tag}_{m.ym}.zst" for m in sorted(valid)]
    else:
        target_paths = build_target_paths()
        relpaths = target_paths[kind]

    paths = [RAW_DIR / r for r in relpaths]
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


# ── Fast sample-only streaming (no stat accumulation) ────────────────────────

# Byte-level pattern to extract created_utc without full JSON parse.
_TS_RE = re.compile(rb'"created_utc"\s*:\s*"?(\d+)"?')


def _reservoir_sample_file(
    path: Path,
    n: int,
    seed: int,
) -> tuple[int, list[dict]]:
    """Reservoir-sample a single .zst file.  Returns ``(in_window_count, reservoir)``.

    Uses two key optimizations:
    - **Lazy parsing**: extracts ``created_utc`` via regex on raw bytes;
      ``orjson.loads`` is called only for the ~n records that enter the reservoir.
    - **Algorithm L** (Li 1994): after the reservoir fills, computes a random
      skip count so most lines are counted but never touched by the RNG.

    This is intentionally a module-level function so it can be pickled for
    ``ProcessPoolExecutor``.
    """
    reservoir: list[dict] = []
    rng = random.Random(seed)
    idx = 0  # in-window record counter
    skip = 0  # records left to skip (Algorithm L)
    w = 0.0  # Algorithm L weight

    dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
    with path.open("rb") as fin:
        with dctx.stream_reader(fin) as zin:
            buf = io.BufferedReader(zin)
            for line in buf:
                # ── Fast timestamp check on raw bytes ──
                m = _TS_RE.search(line)
                if m is None:
                    continue
                ts = int(m.group(1))
                if not (START_EPOCH <= ts < END_EXCLUSIVE_EPOCH):
                    continue

                # ── In-window record ──
                if idx < n:
                    # Filling phase: must parse, store in reservoir.
                    try:
                        reservoir.append(orjson.loads(line))
                    except orjson.JSONDecodeError:
                        continue
                    idx += 1
                    if idx == n:
                        # Transition to Algorithm L skip-ahead.
                        w = math.exp(math.log(rng.random()) / n)
                        skip = int(math.log(rng.random()) / math.log(1 - w))
                else:
                    if skip > 0:
                        skip -= 1
                        idx += 1
                        continue
                    # This record is selected — parse it.
                    try:
                        obj = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        idx += 1
                        continue
                    reservoir[rng.randint(0, n - 1)] = obj
                    idx += 1
                    w *= math.exp(math.log(rng.random()) / n)
                    skip = int(math.log(rng.random()) / math.log(1 - w))

    return idx, reservoir


def _merge_reservoirs(
    results: list[tuple[int, list[dict]]],
    n: int,
    seed: int,
) -> list[dict]:
    """Merge per-file reservoir samples into a single reservoir of size *n*.

    Each sub-reservoir is weighted proportionally by its source population
    so that the final sample is uniform over the union of all files.
    """
    total = sum(count for count, _ in results)
    if total == 0:
        return []

    merged: list[dict] = []
    rng = random.Random(seed)

    for count, reservoir in results:
        # How many slots this file should contribute (probabilistic).
        share = int(round(n * count / total)) if total else 0
        share = min(share, len(reservoir))
        merged.extend(rng.sample(reservoir, share) if share < len(reservoir) else reservoir)

    # Trim or pad to exactly *n*.
    if len(merged) > n:
        merged = rng.sample(merged, n)
    elif len(merged) < n:
        # Fill from the largest reservoir.
        pool = [rec for _, res in results for rec in res if rec not in merged]
        deficit = n - len(merged)
        merged.extend(pool[:deficit])

    return merged[:n]


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


def sample(
    kind: str,
    n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    months: list[tuple[int, int]] | None = None,
    parallel: bool = True,
) -> list[dict]:
    """Return a reservoir sample of *n* in-window records.

    Parameters
    ----------
    months
        Restrict to specific months, e.g. ``[(2022, 10)]`` for October only.
    parallel
        Use one process per .zst file when ``True`` and >1 file is present.
    """
    paths = _resolve_paths(kind, months=months)

    if parallel and len(paths) > 1:
        log.info("Parallel sampling %d files …", len(paths))
        with ProcessPoolExecutor(max_workers=len(paths)) as pool:
            futures = [
                pool.submit(_reservoir_sample_file, p, n, seed + i)
                for i, p in enumerate(paths)
            ]
            results = [f.result() for f in futures]
        reservoir = _merge_reservoirs(results, n, seed)
    else:
        log.info("Sampling %s (%d file(s)) …", kind, len(paths))
        _, reservoir = _reservoir_sample_file(paths[0], n, seed) if len(paths) == 1 else (
            # fallback for non-parallel multi-file
            _stream_analyse(paths, collect_sample=True, sample_n=n, seed=seed)
        )

    log.info("%s: sampled %d records", kind, len(reservoir))
    return reservoir


def _cache_key(kind: str, n: int, seed: int, months: list[tuple[int, int]] | None) -> str:
    """Deterministic short hash for cache filenames."""
    raw = f"{kind}-{n}-{seed}-{sorted(months) if months else 'all'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def sample_dataframe(
    kind: str,
    n: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    months: list[tuple[int, int]] | None = None,
    parallel: bool = True,
    cache: bool = True,
) -> object:
    """Return the reservoir sample as a ``pandas.DataFrame``.

    When *cache* is ``True`` (default), the result is persisted under
    ``output/tables/`` and loaded instantly on subsequent calls.
    Falls back to JSONL if Parquet serialisation fails (mixed-type columns).
    """
    import pandas as pd

    if cache:
        tag = _cache_key(kind, n, seed, months)
        parquet_path = TABLES_DIR / f"sample_{kind}_{tag}.parquet"
        jsonl_path = TABLES_DIR / f"sample_{kind}_{tag}.jsonl"
        for cp in (parquet_path, jsonl_path):
            if cp.exists():
                log.info("Loading cached sample from %s", cp.name)
                if cp.suffix == ".parquet":
                    return pd.read_parquet(cp)
                return pd.read_json(cp, lines=True)

    records = sample(kind, n=n, seed=seed, months=months, parallel=parallel)
    df = pd.DataFrame(records)

    if cache:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(parquet_path, index=False)
            log.info("Cached sample → %s", parquet_path)
        except Exception:
            log.warning(
                "Parquet write failed (mixed types); falling back to JSONL cache."
            )
            df.to_json(jsonl_path, orient="records", lines=True)
            log.info("Cached sample → %s", jsonl_path)

    return df

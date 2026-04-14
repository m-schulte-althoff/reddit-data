"""Comment depth and discursivity analysis for filtered Reddit data.

Computes a hierarchical depth measure per comment:

- Submission: depth 0 (counted separately)
- Direct comment on submission (``parent_id`` starts with ``t3_``): depth 1
- Reply to comment (``parent_id`` starts with ``t1_``): depth 2+

Depth is resolved via a single-pass cascade: when a parent comment has
already been seen its depth is looked up immediately; otherwise the child
is parked in a *pending* dict and resolved once the parent appears.

Aggregated per subreddit per month into ``DepthBucket`` accumulators that
track count, sum, max and a full depth histogram — enough to derive mean
depth, threading ratio, and percentile distributions downstream.
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
class DepthBucket:
    """Running depth statistics for one (subreddit, month) cell."""

    count: int = 0
    depth_sum: int = 0
    max_depth: int = 0
    depth_histogram: Counter[int] = field(default_factory=Counter)

    def add(self, depth: int) -> None:
        """Record one comment at *depth*."""
        self.count += 1
        self.depth_sum += depth
        if depth > self.max_depth:
            self.max_depth = depth
        self.depth_histogram[depth] += 1

    @property
    def mean_depth(self) -> float:
        """Mean comment depth."""
        return self.depth_sum / self.count if self.count else 0.0

    @property
    def threading_ratio(self) -> float:
        """Share of comments at depth >= 2 (replies to other comments)."""
        deep = sum(c for d, c in self.depth_histogram.items() if d >= 2)
        return deep / self.count if self.count else 0.0


@dataclass
class DiscursivityResult:
    """Full result of the discursivity analysis."""

    total_comments: int = 0
    resolved_comments: int = 0
    unresolved_comments: int = 0
    parse_errors: int = 0
    submission_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    buckets: dict[tuple[str, str], DepthBucket] = field(default_factory=dict)

    def sorted_months(self) -> list[str]:
        """All months present in either comments or submissions, sorted."""
        months: set[str] = set()
        for _, m in self.buckets:
            months.add(m)
        for _, m in self.submission_counts:
            months.add(m)
        return sorted(months)

    def sorted_subreddits(self) -> list[str]:
        """Subreddits ordered by total comment count (descending)."""
        subs: Counter[str] = Counter()
        for (sub, _), bucket in self.buckets.items():
            subs[sub] += bucket.count
        return [s for s, _ in subs.most_common()]

    def get_bucket(self, sub: str, month: str) -> DepthBucket:
        """Return (or create) the ``DepthBucket`` for *(sub, month)*."""
        key = (sub, month)
        if key not in self.buckets:
            self.buckets[key] = DepthBucket()
        return self.buckets[key]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _epoch_to_month(epoch: int) -> str:
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


def _stream_zst(path: Path):  # noqa: ANN201
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


def _discover_filtered_paths(kind: str) -> list[Path]:
    """Find filtered .zst files for *kind* in ``PROCESSED_DIR``."""
    paths = sorted(PROCESSED_DIR.glob(f"filter-{kind}-*.jsonl.zst"))
    return [p for p in paths if not p.name.endswith(".progress.json")]


# ── Pending-comment record ───────────────────────────────────────────────────


@dataclass
class _PendingComment:
    """Comment waiting for its parent depth to be resolved."""

    comment_id: str
    subreddit: str
    month: str


# ── Core engine ──────────────────────────────────────────────────────────────


def compute_discursivity(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
) -> DiscursivityResult:
    """Compute comment-depth statistics from filtered data.

    Parameters
    ----------
    comment_paths
        Explicit .zst paths for comments.  ``None`` auto-discovers.
    submission_paths
        Explicit .zst paths for submissions.  ``None`` auto-discovers.
    """
    if comment_paths is None:
        comment_paths = _discover_filtered_paths("comments")
    if submission_paths is None:
        submission_paths = _discover_filtered_paths("submissions")

    result = DiscursivityResult()

    # 1. Count submissions per (subreddit, month).
    for path in submission_paths:
        log.info("Counting submissions in %s …", path.name)
        for obj in _stream_zst(path):
            ts = _extract_created_utc(obj)
            if ts is None:
                continue
            sub = obj.get("subreddit", "unknown")
            month = _epoch_to_month(ts)
            result.submission_counts[(sub, month)] += 1

    # 2. Stream comments — single-pass depth resolution with cascade.
    depth_map: dict[str, int] = {}
    pending: dict[str, list[_PendingComment]] = {}

    for path in comment_paths:
        log.info("Computing comment depths from %s …", path.name)
        for obj in _stream_zst(path):
            result.total_comments += 1

            comment_id: str = obj.get("id", "")
            parent_id: str = obj.get("parent_id", "")
            ts = _extract_created_utc(obj)
            sub: str = obj.get("subreddit", "unknown")
            month = _epoch_to_month(ts) if ts is not None else "unknown"

            if parent_id.startswith("t3_"):
                depth = 1
            elif parent_id.startswith("t1_"):
                parent_key = parent_id[3:]
                if parent_key in depth_map:
                    depth = depth_map[parent_key] + 1
                else:
                    if parent_key not in pending:
                        pending[parent_key] = []
                    pending[parent_key].append(
                        _PendingComment(comment_id, sub, month),
                    )
                    continue
            else:
                result.parse_errors += 1
                continue

            # Resolved — record and cascade.
            depth_map[comment_id] = depth
            result.resolved_comments += 1
            result.get_bucket(sub, month).add(depth)
            _cascade_resolve(comment_id, depth, depth_map, pending, result)

    result.unresolved_comments = result.total_comments - result.resolved_comments - result.parse_errors
    if result.unresolved_comments > 0:
        log.warning(
            "%d comments (%.1f%%) unresolved — parent outside filtered set",
            result.unresolved_comments,
            100 * result.unresolved_comments / result.total_comments
            if result.total_comments
            else 0,
        )

    log.info(
        "Discursivity: %d resolved, %d unresolved, %d parse errors, %d buckets",
        result.resolved_comments,
        result.unresolved_comments,
        result.parse_errors,
        len(result.buckets),
    )
    return result


def _cascade_resolve(
    comment_id: str,
    depth: int,
    depth_map: dict[str, int],
    pending: dict[str, list[_PendingComment]],
    result: DiscursivityResult,
) -> None:
    """Iteratively resolve pending children once their parent is known."""
    stack: list[tuple[str, int]] = [(comment_id, depth)]
    while stack:
        cid, d = stack.pop()
        children = pending.pop(cid, [])
        for child in children:
            child_depth = d + 1
            depth_map[child.comment_id] = child_depth
            result.resolved_comments += 1
            result.get_bucket(child.subreddit, child.month).add(child_depth)
            stack.append((child.comment_id, child_depth))

"""Monthly subreddit panel for downstream WIP analyses."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import TABLES_DIR
from src.discursivity import compute_discursivity, load_discursivity, save_discursivity
from src.helpers import ConcentrationMetrics, _compute_metrics, classify_subreddit, compute_helpers
from src.io_utils import (
    discover_filtered_paths,
    epoch_to_month,
    extract_created_utc,
    fingerprint_hash,
    fingerprint_paths,
    is_deleted_removed,
    iter_month_range,
    months_since,
    safe_divide,
    stream_zst,
)
from src.thread_prep import ThreadPrepConfig, prepare_thread_partitions

log = logging.getLogger(__name__)

PANEL_FILENAME = "community-monthly-panel.csv"
PANEL_METADATA_FILENAME = "community-monthly-panel-metadata.json"
GENAI_REFERENCE_MONTH = "2022-11"
PANEL_CACHE_VERSION = 1


@dataclass
class LengthDistribution:
    """Exact length statistics from a histogram."""

    count: int = 0
    total_length: int = 0
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, text: str) -> None:
        """Record one text value."""
        length = len(text)
        self.count += 1
        self.total_length += length
        self.histogram[length] += 1

    @property
    def mean(self) -> float:
        """Mean text length."""
        return safe_divide(self.total_length, self.count)

    @property
    def median(self) -> float:
        """Exact median text length from the histogram."""
        if self.count == 0:
            return 0.0

        left_rank = (self.count - 1) // 2 + 1
        right_rank = self.count // 2 + 1
        left_value: int | None = None
        right_value: int | None = None
        seen = 0

        for length in sorted(self.histogram):
            seen += self.histogram[length]
            if seen >= left_rank and left_value is None:
                left_value = length
            if seen >= right_rank:
                right_value = length
                break

        return ((left_value or 0) + (right_value or 0)) / 2


@dataclass
class MeanAccumulator:
    """Running mean accumulator."""

    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        """Add one numeric observation."""
        self.count += 1
        self.total += value

    @property
    def mean(self) -> float:
        """Mean value."""
        return safe_divide(self.total, self.count)


@dataclass
class PanelAccumulator:
    """Streaming aggregator for one ``(subreddit, month)`` cell."""

    comments: int = 0
    submissions: int = 0
    unique_comment_authors: set[str] = field(default_factory=set)
    unique_submission_authors: set[str] = field(default_factory=set)
    comment_author_counts: Counter[str] = field(default_factory=Counter)
    comment_lengths: LengthDistribution = field(default_factory=LengthDistribution)
    submission_title_lengths: MeanAccumulator = field(default_factory=MeanAccumulator)
    submission_selftext_lengths: MeanAccumulator = field(default_factory=MeanAccumulator)
    deleted_removed_comments: int = 0
    deleted_removed_submissions: int = 0
    comment_scores: MeanAccumulator = field(default_factory=MeanAccumulator)
    submission_scores: MeanAccumulator = field(default_factory=MeanAccumulator)


@dataclass(frozen=True)
class PartitionedPanelCell:
    """Disk-backed monthly panel cell materialized after shard aggregation."""

    comments: int = 0
    submissions: int = 0
    unique_comment_authors: int = 0
    unique_submission_authors: int = 0
    mean_comment_length: float = 0.0
    median_comment_length: float = 0.0
    mean_submission_title_length: float = 0.0
    mean_submission_selftext_length: float = 0.0
    deleted_removed_comments: int = 0
    deleted_removed_submissions: int = 0
    mean_score_comments: float = 0.0
    mean_score_submissions: float = 0.0


@dataclass(frozen=True)
class PanelRow:
    """Final row written to the monthly panel CSV."""

    subreddit: str
    month: str
    community_type: str
    comments: int
    submissions: int
    comments_per_submission: float
    unique_comment_authors: int
    unique_submission_authors: int
    mean_comment_length: float
    median_comment_length: float
    mean_submission_title_length: float
    mean_submission_selftext_length: float
    deleted_removed_comment_share: float
    deleted_removed_submission_share: float
    mean_score_comments: float
    mean_score_submissions: float
    mean_depth: float
    threading_ratio: float
    max_depth: int
    top1_share: float
    top5_share: float
    hhi: float
    gini: float
    pct1_share: float
    pct9_share: float
    pct90_share: float
    post_genai: int
    months_since_genai: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize in a stable field order for CSV writing."""
        return asdict(self)


@dataclass
class PanelBuildResult:
    """Panel rows plus cache metadata."""

    rows: list[PanelRow]
    metadata: dict[str, Any]


def _normalize_text(value: Any) -> str:
    """Return a string representation for text fields."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _normalize_author(value: Any) -> str | None:
    """Return an author name or ``None`` for deleted/removed placeholders."""
    author = _normalize_text(value)
    if not author or is_deleted_removed(author):
        return None
    return author


def _extract_score(record: dict[str, Any]) -> float | None:
    """Return the numeric score if present."""
    value = record.get("score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell_for(
    cells: dict[tuple[str, str], PanelAccumulator],
    subreddit: str,
    month: str,
) -> PanelAccumulator:
    key = (subreddit, month)
    if key not in cells:
        cells[key] = PanelAccumulator()
    return cells[key]


def _stream_panel_cells(
    comment_paths: list[Path],
    submission_paths: list[Path],
) -> dict[tuple[str, str], PanelAccumulator]:
    """Stream processed files and aggregate all non-depth panel metrics."""
    cells: dict[tuple[str, str], PanelAccumulator] = {}

    for path in submission_paths:
        log.info("Aggregating submissions from %s …", path.name)
        for record in stream_zst(path):
            created_utc = extract_created_utc(record)
            if created_utc is None:
                continue

            subreddit = _normalize_text(record.get("subreddit")) or "unknown"
            month = epoch_to_month(created_utc)
            cell = _cell_for(cells, subreddit, month)
            cell.submissions += 1

            author = _normalize_author(record.get("author"))
            if author is not None:
                cell.unique_submission_authors.add(author)

            title = _normalize_text(record.get("title"))
            selftext = _normalize_text(record.get("selftext"))
            removed = (
                author is None
                or is_deleted_removed(title)
                or is_deleted_removed(selftext)
            )
            if removed:
                cell.deleted_removed_submissions += 1
            else:
                cell.submission_title_lengths.add(float(len(title)))
                cell.submission_selftext_lengths.add(float(len(selftext)))

            score = _extract_score(record)
            if score is not None:
                cell.submission_scores.add(score)

    for path in comment_paths:
        log.info("Aggregating comments from %s …", path.name)
        for record in stream_zst(path):
            created_utc = extract_created_utc(record)
            if created_utc is None:
                continue

            subreddit = _normalize_text(record.get("subreddit")) or "unknown"
            month = epoch_to_month(created_utc)
            cell = _cell_for(cells, subreddit, month)
            cell.comments += 1

            author = _normalize_author(record.get("author"))
            if author is not None:
                cell.unique_comment_authors.add(author)
                cell.comment_author_counts[author] += 1

            body = _normalize_text(record.get("body"))
            if is_deleted_removed(body) or author is None:
                cell.deleted_removed_comments += 1
            else:
                cell.comment_lengths.add(body)

            score = _extract_score(record)
            if score is not None:
                cell.comment_scores.add(score)

    return cells


def _resolve_discursivity(
    comment_paths: list[Path],
    submission_paths: list[Path],
    *,
    cache_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> tuple[Any, bool]:
    """Load a valid discursivity cache or compute it on demand."""
    cached = load_discursivity(comment_paths, submission_paths, cache_dir=cache_dir)
    if cached is not None:
        return cached, True

    result = compute_discursivity(
        comment_paths=comment_paths,
        submission_paths=submission_paths,
        thread_prep=thread_prep,
    )
    if result.resolved_comments > 0:
        save_discursivity(result, comment_paths, submission_paths, out_dir=cache_dir)
    return result, False


def _sources_payload(comment_paths: list[Path], submission_paths: list[Path]) -> dict[str, Any]:
    """Build the current input-source payload."""
    return {
        "version": PANEL_CACHE_VERSION,
        "comment_files": fingerprint_paths(comment_paths),
        "submission_files": fingerprint_paths(submission_paths),
    }


def build_monthly_panel(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    cache_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> PanelBuildResult:
    """Compute the monthly subreddit panel from processed files."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")

    helper_metrics: dict[tuple[str, str], ConcentrationMetrics] | None = None
    partitioned_cells: dict[tuple[str, str], PartitionedPanelCell] | None = None
    cells: dict[tuple[str, str], PanelAccumulator] | None = None

    if thread_prep is not None and thread_prep.enabled:
        partitioned_cells = _stream_panel_cells_partitioned(
            resolved_comment_paths,
            resolved_submission_paths,
            thread_prep=thread_prep,
        )
        helper_result = compute_helpers(
            resolved_comment_paths,
            thread_prep=thread_prep,
        )
        helper_metrics = helper_result.cells
        observed_months = sorted({month for _, month in partitioned_cells})
        subreddits = sorted({subreddit for subreddit, _ in partitioned_cells}, key=str.casefold)
    else:
        cells = _stream_panel_cells(resolved_comment_paths, resolved_submission_paths)
        observed_months = sorted({month for _, month in cells})
        subreddits = sorted({subreddit for subreddit, _ in cells}, key=str.casefold)

    discursivity, used_discursivity_cache = _resolve_discursivity(
        resolved_comment_paths,
        resolved_submission_paths,
        cache_dir=cache_dir,
        thread_prep=thread_prep,
    )

    rows: list[PanelRow] = []
    months_covered = (
        iter_month_range(observed_months[0], observed_months[-1])
        if observed_months
        else []
    )

    for subreddit in subreddits:
        for month in months_covered:
            depth_bucket = discursivity.buckets.get((subreddit, month))
            months_relative = months_since(month, reference=GENAI_REFERENCE_MONTH)

            if partitioned_cells is not None and helper_metrics is not None:
                base_cell = partitioned_cells.get((subreddit, month), PartitionedPanelCell())
                concentration = helper_metrics.get((subreddit, month), ConcentrationMetrics())
                comments = base_cell.comments
                submissions = base_cell.submissions
                unique_comment_authors = base_cell.unique_comment_authors
                unique_submission_authors = base_cell.unique_submission_authors
                mean_comment_length = base_cell.mean_comment_length
                median_comment_length = base_cell.median_comment_length
                mean_submission_title_length = base_cell.mean_submission_title_length
                mean_submission_selftext_length = base_cell.mean_submission_selftext_length
                deleted_removed_comment_share = safe_divide(
                    base_cell.deleted_removed_comments,
                    comments,
                )
                deleted_removed_submission_share = safe_divide(
                    base_cell.deleted_removed_submissions,
                    submissions,
                )
                mean_score_comments = base_cell.mean_score_comments
                mean_score_submissions = base_cell.mean_score_submissions
            else:
                cell = cells.get((subreddit, month), PanelAccumulator()) if cells is not None else PanelAccumulator()
                concentration = _compute_metrics(cell.comment_author_counts)
                comments = cell.comments
                submissions = cell.submissions
                unique_comment_authors = len(cell.unique_comment_authors)
                unique_submission_authors = len(cell.unique_submission_authors)
                mean_comment_length = cell.comment_lengths.mean
                median_comment_length = cell.comment_lengths.median
                mean_submission_title_length = cell.submission_title_lengths.mean
                mean_submission_selftext_length = cell.submission_selftext_lengths.mean
                deleted_removed_comment_share = safe_divide(
                    cell.deleted_removed_comments,
                    cell.comments,
                )
                deleted_removed_submission_share = safe_divide(
                    cell.deleted_removed_submissions,
                    cell.submissions,
                )
                mean_score_comments = cell.comment_scores.mean
                mean_score_submissions = cell.submission_scores.mean

            rows.append(
                PanelRow(
                    subreddit=subreddit,
                    month=month,
                    community_type=classify_subreddit(subreddit),
                    comments=comments,
                    submissions=submissions,
                    comments_per_submission=safe_divide(comments, submissions),
                    unique_comment_authors=unique_comment_authors,
                    unique_submission_authors=unique_submission_authors,
                    mean_comment_length=mean_comment_length,
                    median_comment_length=median_comment_length,
                    mean_submission_title_length=mean_submission_title_length,
                    mean_submission_selftext_length=mean_submission_selftext_length,
                    deleted_removed_comment_share=deleted_removed_comment_share,
                    deleted_removed_submission_share=deleted_removed_submission_share,
                    mean_score_comments=mean_score_comments,
                    mean_score_submissions=mean_score_submissions,
                    mean_depth=depth_bucket.mean_depth if depth_bucket is not None else 0.0,
                    threading_ratio=(
                        depth_bucket.threading_ratio if depth_bucket is not None else 0.0
                    ),
                    max_depth=depth_bucket.max_depth if depth_bucket is not None else 0,
                    top1_share=concentration.top1_share,
                    top5_share=concentration.top5_share,
                    hhi=concentration.hhi,
                    gini=concentration.gini,
                    pct1_share=concentration.pct1_share,
                    pct9_share=concentration.pct9_share,
                    pct90_share=concentration.pct90_share,
                    post_genai=1 if months_relative >= 0 else 0,
                    months_since_genai=months_relative,
                ),
            )

    sources = _sources_payload(resolved_comment_paths, resolved_submission_paths)
    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "fingerprint": fingerprint_hash(sources),
        "sources": sources,
        "n_rows": len(rows),
        "n_subreddits": len(subreddits),
        "months_covered": months_covered,
        "discursivity_cache_used": used_discursivity_cache,
        "thread_prep_partitions": thread_prep.partitions if thread_prep is not None and thread_prep.enabled else 0,
        "text_metrics_exclude_deleted_removed": True,
    }
    return PanelBuildResult(rows=rows, metadata=metadata)


def save_monthly_panel(
    result: PanelBuildResult,
    *,
    tables_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write panel outputs to ``tables_dir``."""
    base_dir = tables_dir or TABLES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / PANEL_FILENAME
    metadata_path = base_dir / PANEL_METADATA_FILENAME

    fieldnames = [field_info.name for field_info in fields(PanelRow)]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.to_dict())

    metadata_path.write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info("Wrote %s", csv_path)
    log.info("Wrote %s", metadata_path)
    return csv_path, metadata_path


def load_panel_cache(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return cache metadata when the panel outputs match current inputs."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    base_dir = tables_dir or TABLES_DIR
    csv_path = base_dir / PANEL_FILENAME
    metadata_path = base_dir / PANEL_METADATA_FILENAME

    if not csv_path.exists() or not metadata_path.exists():
        return None

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read panel metadata: %s", exc)
        return None

    current_sources = _sources_payload(resolved_comment_paths, resolved_submission_paths)
    if payload.get("fingerprint") != fingerprint_hash(current_sources):
        log.info("Monthly panel cache stale (fingerprint mismatch) — will recompute")
        return None

    return payload


def ensure_monthly_panel(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Ensure the panel outputs exist and are fresh for the current inputs."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    base_dir = tables_dir or TABLES_DIR

    metadata = load_panel_cache(
        resolved_comment_paths,
        resolved_submission_paths,
        tables_dir=base_dir,
    )
    csv_path = base_dir / PANEL_FILENAME
    metadata_path = base_dir / PANEL_METADATA_FILENAME
    if metadata is not None:
        log.info("Using cached monthly panel at %s", csv_path)
        return csv_path, metadata_path, metadata

    result = build_monthly_panel(
        comment_paths=resolved_comment_paths,
        submission_paths=resolved_submission_paths,
        cache_dir=base_dir,
        thread_prep=thread_prep,
    )
    csv_path, metadata_path = save_monthly_panel(result, tables_dir=base_dir)
    return csv_path, metadata_path, result.metadata


@dataclass
class _SqlPanelAccumulator:
    """Running monthly totals flushed into a SQLite-backed panel cache."""

    comments: int = 0
    submissions: int = 0
    deleted_removed_comments: int = 0
    deleted_removed_submissions: int = 0
    comment_length_total: int = 0
    comment_length_count: int = 0
    submission_title_length_total: int = 0
    submission_title_length_count: int = 0
    submission_selftext_length_total: int = 0
    submission_selftext_length_count: int = 0
    comment_score_total: float = 0.0
    comment_score_count: int = 0
    submission_score_total: float = 0.0
    submission_score_count: int = 0


def _stream_panel_cells_partitioned(
    comment_paths: list[Path],
    submission_paths: list[Path],
    *,
    thread_prep: ThreadPrepConfig,
) -> dict[tuple[str, str], PartitionedPanelCell]:
    """Aggregate panel base metrics via submission-hash shards and SQLite."""
    partitioned = prepare_thread_partitions(comment_paths, submission_paths, config=thread_prep)
    sqlite_path = partitioned.root_dir / "panel-base.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE cell_metrics (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            comments INTEGER NOT NULL DEFAULT 0,
            submissions INTEGER NOT NULL DEFAULT 0,
            deleted_removed_comments INTEGER NOT NULL DEFAULT 0,
            deleted_removed_submissions INTEGER NOT NULL DEFAULT 0,
            comment_length_total INTEGER NOT NULL DEFAULT 0,
            comment_length_count INTEGER NOT NULL DEFAULT 0,
            submission_title_length_total INTEGER NOT NULL DEFAULT 0,
            submission_title_length_count INTEGER NOT NULL DEFAULT 0,
            submission_selftext_length_total INTEGER NOT NULL DEFAULT 0,
            submission_selftext_length_count INTEGER NOT NULL DEFAULT 0,
            comment_score_total REAL NOT NULL DEFAULT 0,
            comment_score_count INTEGER NOT NULL DEFAULT 0,
            submission_score_total REAL NOT NULL DEFAULT 0,
            submission_score_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (subreddit, month)
        );

        CREATE TABLE comment_lengths (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            length INTEGER NOT NULL,
            length_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (subreddit, month, length)
        );

        CREATE TABLE comment_authors (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            PRIMARY KEY (subreddit, month, author)
        );

        CREATE TABLE submission_authors (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            PRIMARY KEY (subreddit, month, author)
        );
        """,
    )

    for path in partitioned.submission_partitions:
        _aggregate_submission_partition(conn, path)
    for path in partitioned.comment_partitions:
        _aggregate_comment_partition(conn, path)

    cells = _load_partitioned_panel_cells(conn)
    conn.close()
    return cells


def _aggregate_submission_partition(conn: sqlite3.Connection, path: Path) -> None:
    metrics: dict[tuple[str, str], _SqlPanelAccumulator] = {}
    author_rows: set[tuple[str, str, str]] = set()

    log.info("Aggregating partitioned submissions from %s …", path.name)
    for record in stream_zst(path):
        created_utc = extract_created_utc(record)
        if created_utc is None:
            continue

        subreddit = _normalize_text(record.get("subreddit")) or "unknown"
        month = epoch_to_month(created_utc)
        cell = metrics.setdefault((subreddit, month), _SqlPanelAccumulator())
        cell.submissions += 1

        author = _normalize_author(record.get("author"))
        if author is not None:
            author_rows.add((subreddit, month, author))

        title = _normalize_text(record.get("title"))
        selftext = _normalize_text(record.get("selftext"))
        removed = author is None or is_deleted_removed(title) or is_deleted_removed(selftext)
        if removed:
            cell.deleted_removed_submissions += 1
        else:
            cell.submission_title_length_total += len(title)
            cell.submission_title_length_count += 1
            cell.submission_selftext_length_total += len(selftext)
            cell.submission_selftext_length_count += 1

        score = _extract_score(record)
        if score is not None:
            cell.submission_score_total += score
            cell.submission_score_count += 1

    _flush_panel_metrics(conn, metrics)
    if author_rows:
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO submission_authors (subreddit, month, author) VALUES (?, ?, ?)",
                list(author_rows),
            )


def _aggregate_comment_partition(conn: sqlite3.Connection, path: Path) -> None:
    metrics: dict[tuple[str, str], _SqlPanelAccumulator] = {}
    author_rows: set[tuple[str, str, str]] = set()
    length_rows: Counter[tuple[str, str, int]] = Counter()

    log.info("Aggregating partitioned comments from %s …", path.name)
    for record in stream_zst(path):
        created_utc = extract_created_utc(record)
        if created_utc is None:
            continue

        subreddit = _normalize_text(record.get("subreddit")) or "unknown"
        month = epoch_to_month(created_utc)
        cell = metrics.setdefault((subreddit, month), _SqlPanelAccumulator())
        cell.comments += 1

        author = _normalize_author(record.get("author"))
        if author is not None:
            author_rows.add((subreddit, month, author))

        body = _normalize_text(record.get("body"))
        if is_deleted_removed(body) or author is None:
            cell.deleted_removed_comments += 1
        else:
            length = len(body)
            cell.comment_length_total += length
            cell.comment_length_count += 1
            length_rows[(subreddit, month, length)] += 1

        score = _extract_score(record)
        if score is not None:
            cell.comment_score_total += score
            cell.comment_score_count += 1

    _flush_panel_metrics(conn, metrics)
    with conn:
        if author_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO comment_authors (subreddit, month, author) VALUES (?, ?, ?)",
                list(author_rows),
            )
        if length_rows:
            conn.executemany(
                """
                INSERT INTO comment_lengths (subreddit, month, length, length_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subreddit, month, length) DO UPDATE SET
                    length_count = comment_lengths.length_count + excluded.length_count
                """,
                [(*key, count) for key, count in length_rows.items()],
            )


def _flush_panel_metrics(
    conn: sqlite3.Connection,
    metrics: dict[tuple[str, str], _SqlPanelAccumulator],
) -> None:
    if not metrics:
        return
    with conn:
        conn.executemany(
            """
            INSERT INTO cell_metrics (
                subreddit,
                month,
                comments,
                submissions,
                deleted_removed_comments,
                deleted_removed_submissions,
                comment_length_total,
                comment_length_count,
                submission_title_length_total,
                submission_title_length_count,
                submission_selftext_length_total,
                submission_selftext_length_count,
                comment_score_total,
                comment_score_count,
                submission_score_total,
                submission_score_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subreddit, month) DO UPDATE SET
                comments = cell_metrics.comments + excluded.comments,
                submissions = cell_metrics.submissions + excluded.submissions,
                deleted_removed_comments = cell_metrics.deleted_removed_comments + excluded.deleted_removed_comments,
                deleted_removed_submissions = cell_metrics.deleted_removed_submissions + excluded.deleted_removed_submissions,
                comment_length_total = cell_metrics.comment_length_total + excluded.comment_length_total,
                comment_length_count = cell_metrics.comment_length_count + excluded.comment_length_count,
                submission_title_length_total = cell_metrics.submission_title_length_total + excluded.submission_title_length_total,
                submission_title_length_count = cell_metrics.submission_title_length_count + excluded.submission_title_length_count,
                submission_selftext_length_total = cell_metrics.submission_selftext_length_total + excluded.submission_selftext_length_total,
                submission_selftext_length_count = cell_metrics.submission_selftext_length_count + excluded.submission_selftext_length_count,
                comment_score_total = cell_metrics.comment_score_total + excluded.comment_score_total,
                comment_score_count = cell_metrics.comment_score_count + excluded.comment_score_count,
                submission_score_total = cell_metrics.submission_score_total + excluded.submission_score_total,
                submission_score_count = cell_metrics.submission_score_count + excluded.submission_score_count
            """,
            [
                (
                    subreddit,
                    month,
                    cell.comments,
                    cell.submissions,
                    cell.deleted_removed_comments,
                    cell.deleted_removed_submissions,
                    cell.comment_length_total,
                    cell.comment_length_count,
                    cell.submission_title_length_total,
                    cell.submission_title_length_count,
                    cell.submission_selftext_length_total,
                    cell.submission_selftext_length_count,
                    cell.comment_score_total,
                    cell.comment_score_count,
                    cell.submission_score_total,
                    cell.submission_score_count,
                )
                for (subreddit, month), cell in metrics.items()
            ],
        )


def _load_partitioned_panel_cells(conn: sqlite3.Connection) -> dict[tuple[str, str], PartitionedPanelCell]:
    comment_author_counts = {
        (str(row[0]), str(row[1])): int(row[2])
        for row in conn.execute(
            "SELECT subreddit, month, COUNT(*) FROM comment_authors GROUP BY subreddit, month",
        )
    }
    submission_author_counts = {
        (str(row[0]), str(row[1])): int(row[2])
        for row in conn.execute(
            "SELECT subreddit, month, COUNT(*) FROM submission_authors GROUP BY subreddit, month",
        )
    }
    comment_medians = _load_comment_length_medians(conn)

    cells: dict[tuple[str, str], PartitionedPanelCell] = {}
    for row in conn.execute(
        """
        SELECT
            subreddit,
            month,
            comments,
            submissions,
            deleted_removed_comments,
            deleted_removed_submissions,
            comment_length_total,
            comment_length_count,
            submission_title_length_total,
            submission_title_length_count,
            submission_selftext_length_total,
            submission_selftext_length_count,
            comment_score_total,
            comment_score_count,
            submission_score_total,
            submission_score_count
        FROM cell_metrics
        ORDER BY subreddit, month
        """,
    ):
        subreddit = str(row["subreddit"])
        month = str(row["month"])
        key = (subreddit, month)
        cells[key] = PartitionedPanelCell(
            comments=int(row["comments"]),
            submissions=int(row["submissions"]),
            unique_comment_authors=comment_author_counts.get(key, 0),
            unique_submission_authors=submission_author_counts.get(key, 0),
            mean_comment_length=safe_divide(int(row["comment_length_total"]), int(row["comment_length_count"])),
            median_comment_length=comment_medians.get(key, 0.0),
            mean_submission_title_length=safe_divide(
                int(row["submission_title_length_total"]),
                int(row["submission_title_length_count"]),
            ),
            mean_submission_selftext_length=safe_divide(
                int(row["submission_selftext_length_total"]),
                int(row["submission_selftext_length_count"]),
            ),
            deleted_removed_comments=int(row["deleted_removed_comments"]),
            deleted_removed_submissions=int(row["deleted_removed_submissions"]),
            mean_score_comments=safe_divide(float(row["comment_score_total"]), int(row["comment_score_count"])),
            mean_score_submissions=safe_divide(float(row["submission_score_total"]), int(row["submission_score_count"])),
        )
    return cells


def _load_comment_length_medians(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    medians: dict[tuple[str, str], float] = {}
    current_key: tuple[str, str] | None = None
    current_rows: list[tuple[int, int]] = []

    for row in conn.execute(
        "SELECT subreddit, month, length, length_count FROM comment_lengths ORDER BY subreddit, month, length",
    ):
        key = (str(row[0]), str(row[1]))
        if current_key is None:
            current_key = key
        if key != current_key:
            medians[current_key] = _median_from_histogram_rows(current_rows)
            current_key = key
            current_rows = []
        current_rows.append((int(row[2]), int(row[3])))

    if current_key is not None:
        medians[current_key] = _median_from_histogram_rows(current_rows)
    return medians


def _median_from_histogram_rows(rows: list[tuple[int, int]]) -> float:
    total = sum(count for _, count in rows)
    if total == 0:
        return 0.0

    left_rank = (total - 1) // 2 + 1
    right_rank = total // 2 + 1
    left_value: int | None = None
    right_value: int | None = None
    seen = 0

    for length, count in rows:
        seen += count
        if seen >= left_rank and left_value is None:
            left_value = length
        if seen >= right_rank:
            right_value = length
            break

    return ((left_value or 0) + (right_value or 0)) / 2
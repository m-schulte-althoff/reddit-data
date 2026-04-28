"""Monthly subreddit panel for downstream WIP analyses."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import TABLES_DIR
from src.discursivity import compute_discursivity, load_discursivity, save_discursivity
from src.helpers import _compute_metrics, classify_subreddit
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
) -> tuple[Any, bool]:
    """Load a valid discursivity cache or compute it on demand."""
    cached = load_discursivity(comment_paths, submission_paths, cache_dir=cache_dir)
    if cached is not None:
        return cached, True

    result = compute_discursivity(comment_paths=comment_paths, submission_paths=submission_paths)
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
) -> PanelBuildResult:
    """Compute the monthly subreddit panel from processed files."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")

    cells = _stream_panel_cells(resolved_comment_paths, resolved_submission_paths)
    discursivity, used_discursivity_cache = _resolve_discursivity(
        resolved_comment_paths,
        resolved_submission_paths,
        cache_dir=cache_dir,
    )

    rows: list[PanelRow] = []
    observed_months = sorted({month for _, month in cells})
    months_covered = (
        iter_month_range(observed_months[0], observed_months[-1])
        if observed_months
        else []
    )
    subreddits = sorted({subreddit for subreddit, _ in cells}, key=str.casefold)

    for subreddit in subreddits:
        for month in months_covered:
            cell = cells.get((subreddit, month), PanelAccumulator())
            depth_bucket = discursivity.buckets.get((subreddit, month))
            concentration = _compute_metrics(cell.comment_author_counts)
            months_relative = months_since(month, reference=GENAI_REFERENCE_MONTH)

            rows.append(
                PanelRow(
                    subreddit=subreddit,
                    month=month,
                    community_type=classify_subreddit(subreddit),
                    comments=cell.comments,
                    submissions=cell.submissions,
                    comments_per_submission=safe_divide(cell.comments, cell.submissions),
                    unique_comment_authors=len(cell.unique_comment_authors),
                    unique_submission_authors=len(cell.unique_submission_authors),
                    mean_comment_length=cell.comment_lengths.mean,
                    median_comment_length=cell.comment_lengths.median,
                    mean_submission_title_length=cell.submission_title_lengths.mean,
                    mean_submission_selftext_length=cell.submission_selftext_lengths.mean,
                    deleted_removed_comment_share=safe_divide(
                        cell.deleted_removed_comments,
                        cell.comments,
                    ),
                    deleted_removed_submission_share=safe_divide(
                        cell.deleted_removed_submissions,
                        cell.submissions,
                    ),
                    mean_score_comments=cell.comment_scores.mean,
                    mean_score_submissions=cell.submission_scores.mean,
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

    fieldnames = list(PanelRow.__dataclass_fields__.keys())
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
    )
    csv_path, metadata_path = save_monthly_panel(result, tables_dir=base_dir)
    return csv_path, metadata_path, result.metadata
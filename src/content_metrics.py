"""Simple effort, support, and information-proxy content metrics."""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, TABLES_DIR
from src.io_utils import discover_filtered_paths, epoch_to_month, extract_created_utc, fingerprint_hash, fingerprint_paths, is_deleted_removed, stream_zst
from src.panel import ensure_monthly_panel
from src.thread_prep import ThreadPrepConfig

log = logging.getLogger(__name__)

CONTENT_METRICS_FILENAME = "content-metrics-monthly.csv"
CONTENT_METRICS_METADATA_FILENAME = "content-metrics-metadata.json"
CONTENT_METRICS_CACHE_VERSION = 1

QUESTION_RE = re.compile(r"\?|\b(how|what|why|should i|anyone|experience)\b", re.IGNORECASE)
EXPERIENCE_RE = re.compile(r"\b(i have|my doctor|my symptoms|diagnosed|anyone else|experience)\b", re.IGNORECASE)
SUPPORT_RE = re.compile(r"\b(sorry|hope|hug|support|feel|you are not alone|take care)\b", re.IGNORECASE)
MEDICAL_RE = re.compile(r"\b(doctor|physician|er|urgent care|not medical advice|ask your doctor)\b", re.IGNORECASE)


@dataclass
class LengthDistribution:
    """Exact length distribution for median and mean statistics."""

    count: int = 0
    total_length: int = 0
    histogram: Counter[int] = field(default_factory=Counter)

    def add(self, text: str) -> None:
        length = len(text)
        self.count += 1
        self.total_length += length
        self.histogram[length] += 1

    @property
    def mean(self) -> float:
        return self.total_length / self.count if self.count else 0.0

    @property
    def median(self) -> float:
        if self.count == 0:
            return 0.0
        left_rank = (self.count - 1) // 2 + 1
        right_rank = self.count // 2 + 1
        seen = 0
        left_value: int | None = None
        right_value: int | None = None
        for length in sorted(self.histogram):
            seen += self.histogram[length]
            if seen >= left_rank and left_value is None:
                left_value = length
            if seen >= right_rank:
                right_value = length
                break
        return ((left_value or 0) + (right_value or 0)) / 2


@dataclass
class TextAccumulator:
    """Streaming text metrics for one channel and subreddit-month cell."""

    lengths: LengthDistribution = field(default_factory=LengthDistribution)
    very_short_count: int = 0
    long_count: int = 0
    question_count: int = 0
    experience_count: int = 0
    support_count: int = 0
    medical_count: int = 0
    question_mark_count: int = 0
    punctuation_chars: int = 0
    sentence_count: int = 0
    word_count: int = 0

    def add(self, text: str) -> None:
        self.lengths.add(text)
        length = len(text)
        if length < 20:
            self.very_short_count += 1
        if length > 500:
            self.long_count += 1
        if QUESTION_RE.search(text):
            self.question_count += 1
        if EXPERIENCE_RE.search(text):
            self.experience_count += 1
        if SUPPORT_RE.search(text):
            self.support_count += 1
        if MEDICAL_RE.search(text):
            self.medical_count += 1
        if "?" in text:
            self.question_mark_count += 1
        self.punctuation_chars += sum(1 for char in text if char in string.punctuation)
        sentences = [segment for segment in re.split(r"[.!?]+", text) if segment.strip()]
        if sentences:
            self.sentence_count += len(sentences)
            self.word_count += sum(len(sentence.split()) for sentence in sentences)


@dataclass
class ContentMetricsArtifacts:
    """Written outputs plus the monthly content-metrics table."""

    monthly: pd.DataFrame
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    metadata: dict[str, Any]


def run_content_metrics_analysis(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> ContentMetricsArtifacts:
    """Compute and cache simple monthly content metrics."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    table_paths = {
        "monthly": out_tables / CONTENT_METRICS_FILENAME,
        "metadata": out_tables / CONTENT_METRICS_METADATA_FILENAME,
    }
    figure_paths = {
        "length": out_figures / "content-length-health-vs-general.svg",
        "question": out_figures / "content-question-share-health-vs-general.svg",
        "experience": out_figures / "content-experience-share-health-vs-general.svg",
        "support": out_figures / "content-support-share-health-vs-general.svg",
    }

    metadata = load_content_metrics_cache(
        resolved_comment_paths,
        resolved_submission_paths,
        tables_dir=out_tables,
        figures_dir=out_figures,
    )
    if metadata is not None:
        return ContentMetricsArtifacts(
            monthly=pd.read_csv(table_paths["monthly"]),
            table_paths=table_paths,
            figure_paths=figure_paths,
            metadata=metadata,
        )

    panel_path, _, _ = ensure_monthly_panel(
        comment_paths=resolved_comment_paths,
        submission_paths=resolved_submission_paths,
        tables_dir=out_tables,
        thread_prep=thread_prep,
    )
    panel = pd.read_csv(panel_path)
    monthly = panel[["subreddit", "month", "community_type", "comments", "submissions"]].copy()

    comment_metrics = _aggregate_text_metrics(resolved_comment_paths, kind="comment")
    submission_metrics = _aggregate_text_metrics(resolved_submission_paths, kind="submission")

    monthly = _merge_channel_metrics(monthly, comment_metrics, prefix="comment")
    monthly = _merge_channel_metrics(monthly, submission_metrics, prefix="submission")
    monthly = monthly.sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)

    monthly.to_csv(table_paths["monthly"], index=False)
    _plot_two_panel_trend(monthly, "comment_mean_length", "submission_mean_length", figure_paths["length"], "Mean text length")
    _plot_two_panel_trend(monthly, "comment_question_share", "submission_question_share", figure_paths["question"], "Question-seeking share")
    _plot_two_panel_trend(monthly, "comment_experience_share", "submission_experience_share", figure_paths["experience"], "Experience-sharing share")
    _plot_two_panel_trend(monthly, "comment_support_share", "submission_support_share", figure_paths["support"], "Support-language share")

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sources": _source_payload(resolved_comment_paths, resolved_submission_paths),
        "fingerprint": fingerprint_hash(_source_payload(resolved_comment_paths, resolved_submission_paths)),
        "n_rows": int(len(monthly)),
    }
    table_paths["metadata"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return ContentMetricsArtifacts(monthly=monthly, table_paths=table_paths, figure_paths=figure_paths, metadata=payload)


def load_content_metrics_cache(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return cache metadata when content-metrics outputs are current."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    table_path = out_tables / CONTENT_METRICS_FILENAME
    metadata_path = out_tables / CONTENT_METRICS_METADATA_FILENAME
    figure_paths = [
        out_figures / "content-length-health-vs-general.svg",
        out_figures / "content-question-share-health-vs-general.svg",
        out_figures / "content-experience-share-health-vs-general.svg",
        out_figures / "content-support-share-health-vs-general.svg",
    ]
    if not table_path.exists() or not metadata_path.exists() or not all(path.exists() for path in figure_paths):
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint_hash(_source_payload(resolved_comment_paths, resolved_submission_paths)):
        return None
    return payload


def _source_payload(comment_paths: list[Path], submission_paths: list[Path]) -> dict[str, Any]:
    return {
        "version": CONTENT_METRICS_CACHE_VERSION,
        "comment_files": fingerprint_paths(comment_paths),
        "submission_files": fingerprint_paths(submission_paths),
    }


def _aggregate_text_metrics(paths: list[Path], *, kind: str) -> dict[tuple[str, str], TextAccumulator]:
    metrics: dict[tuple[str, str], TextAccumulator] = {}
    for path in paths:
        log.info("Aggregating %s content metrics from %s …", kind, path.name)
        for record in stream_zst(path):
            created_utc = extract_created_utc(record)
            if created_utc is None:
                continue
            subreddit = str(record.get("subreddit", "unknown"))
            month = epoch_to_month(created_utc)
            key = (subreddit, month)
            if key not in metrics:
                metrics[key] = TextAccumulator()

            if kind == "submission":
                text = f"{record.get('title', '')} {record.get('selftext', '')}".strip()
            else:
                text = str(record.get("body", ""))
            if not text or is_deleted_removed(text):
                continue
            metrics[key].add(text)
    return metrics


def _merge_channel_metrics(
    monthly: pd.DataFrame,
    metrics: dict[tuple[str, str], TextAccumulator],
    *,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    for (subreddit, month), accumulator in sorted(metrics.items()):
        count = accumulator.lengths.count
        rows.append({
            "subreddit": subreddit,
            "month": month,
            f"{prefix}_text_count": count,
            f"{prefix}_mean_length": accumulator.lengths.mean,
            f"{prefix}_median_length": accumulator.lengths.median,
            f"{prefix}_very_short_share": accumulator.very_short_count / count if count else 0.0,
            f"{prefix}_long_share": accumulator.long_count / count if count else 0.0,
            f"{prefix}_mean_sentence_length": accumulator.word_count / accumulator.sentence_count if accumulator.sentence_count else 0.0,
            f"{prefix}_punctuation_density": accumulator.punctuation_chars / accumulator.lengths.total_length if accumulator.lengths.total_length else 0.0,
            f"{prefix}_question_mark_share": accumulator.question_mark_count / count if count else 0.0,
            f"{prefix}_question_share": accumulator.question_count / count if count else 0.0,
            f"{prefix}_experience_share": accumulator.experience_count / count if count else 0.0,
            f"{prefix}_support_share": accumulator.support_count / count if count else 0.0,
            f"{prefix}_medical_share": accumulator.medical_count / count if count else 0.0,
        })

    metric_frame = pd.DataFrame(rows)
    if metric_frame.empty:
        for column in [
            f"{prefix}_text_count",
            f"{prefix}_mean_length",
            f"{prefix}_median_length",
            f"{prefix}_very_short_share",
            f"{prefix}_long_share",
            f"{prefix}_mean_sentence_length",
            f"{prefix}_punctuation_density",
            f"{prefix}_question_mark_share",
            f"{prefix}_question_share",
            f"{prefix}_experience_share",
            f"{prefix}_support_share",
            f"{prefix}_medical_share",
        ]:
            monthly[column] = 0.0
        return monthly

    merged = monthly.merge(metric_frame, on=["subreddit", "month"], how="left")
    fill_columns = [column for column in merged.columns if column.startswith(f"{prefix}_")]
    merged[fill_columns] = merged[fill_columns].fillna(0.0)
    return merged


def _plot_two_panel_trend(
    frame: pd.DataFrame,
    comment_column: str,
    submission_column: str,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    _plot_metric_by_type(frame, comment_column, axes[0], f"Comments: {title}")
    _plot_metric_by_type(frame, submission_column, axes[1], f"Submissions: {title}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_by_type(frame: pd.DataFrame, column: str, ax: plt.Axes, title: str) -> None:
    plot_frame = (
        frame.loc[frame["community_type"].isin(["health", "general"])]
        .groupby(["month", "community_type"], as_index=False)[column]
        .mean()
    )
    styles = {
        "health": {"label": "Health", "color": "#1b9e77", "marker": "o"},
        "general": {"label": "General", "color": "#d95f02", "marker": "s"},
    }
    for community_type in ["health", "general"]:
        subset = plot_frame.loc[plot_frame["community_type"] == community_type]
        if subset.empty:
            continue
        style = styles[community_type]
        ax.plot(
            subset["month"],
            subset[column],
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            label=style["label"],
        )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)
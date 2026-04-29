"""Descriptive AI-mention trends in comments and submissions."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, TABLES_DIR
from src.io_utils import discover_filtered_paths, epoch_to_month, extract_created_utc, fingerprint_hash, fingerprint_paths, stream_zst
from src.panel import ensure_monthly_panel
from src.thread_prep import ThreadPrepConfig

log = logging.getLogger(__name__)

AI_MENTIONS_FILENAME = "ai-mentions-monthly.csv"
AI_MENTIONS_METADATA_FILENAME = "ai-mentions-metadata.json"
AI_MENTIONS_CACHE_VERSION = 1
AI_MENTION_PATTERN = re.compile(
    r"\b(?:chatgpt|gpt(?:-3|-4)?|openai|llm|large language model|generative ai|ai generated|artificial intelligence|bard|gemini|claude|copilot)\b",
    flags=re.IGNORECASE,
)


@dataclass
class AiMentionsArtifacts:
    """Written outputs plus the monthly AI-mention panel."""

    monthly: pd.DataFrame
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    metadata: dict[str, Any]


def run_ai_mentions_analysis(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> AiMentionsArtifacts:
    """Compute monthly AI-mention counts and write tables/figures."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    table_paths = {
        "monthly": out_tables / AI_MENTIONS_FILENAME,
        "metadata": out_tables / AI_MENTIONS_METADATA_FILENAME,
    }
    figure_paths = {
        "comments": out_figures / "ai-mentions-health-vs-general-comments.svg",
        "submissions": out_figures / "ai-mentions-health-vs-general-submissions.svg",
        "top_subreddits": out_figures / "ai-mentions-top-subreddits.svg",
    }

    metadata = load_ai_mentions_cache(
        resolved_comment_paths,
        resolved_submission_paths,
        tables_dir=out_tables,
        figures_dir=out_figures,
    )
    if metadata is not None:
        return AiMentionsArtifacts(
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
    monthly["ai_mention_comments"] = 0
    monthly["ai_mention_submissions"] = 0

    comment_counts = _count_ai_mentions(resolved_comment_paths, field="body")
    submission_counts = _count_ai_mentions(resolved_submission_paths, field="submission")

    monthly = monthly.merge(
        pd.DataFrame(comment_counts, columns=["subreddit", "month", "ai_mention_comments"]),
        on=["subreddit", "month"],
        how="left",
        suffixes=("", "_new"),
    )
    monthly["ai_mention_comments"] = monthly["ai_mention_comments_new"].fillna(monthly["ai_mention_comments"])
    monthly = monthly.drop(columns=["ai_mention_comments_new"])

    monthly = monthly.merge(
        pd.DataFrame(submission_counts, columns=["subreddit", "month", "ai_mention_submissions"]),
        on=["subreddit", "month"],
        how="left",
        suffixes=("", "_new"),
    )
    monthly["ai_mention_submissions"] = monthly["ai_mention_submissions_new"].fillna(monthly["ai_mention_submissions"])
    monthly = monthly.drop(columns=["ai_mention_submissions_new"])

    monthly["ai_mention_comment_share"] = monthly.apply(
        lambda row: row["ai_mention_comments"] / row["comments"] if row["comments"] else 0.0,
        axis=1,
    )
    monthly["ai_mention_submission_share"] = monthly.apply(
        lambda row: row["ai_mention_submissions"] / row["submissions"] if row["submissions"] else 0.0,
        axis=1,
    )
    monthly = monthly.sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)

    monthly.to_csv(table_paths["monthly"], index=False)
    _plot_type_trends(monthly, "ai_mention_comments", figure_paths["comments"], "AI-mention comments")
    _plot_type_trends(monthly, "ai_mention_submissions", figure_paths["submissions"], "AI-mention submissions")
    _plot_top_subreddits(monthly, figure_paths["top_subreddits"])

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sources": _source_payload(resolved_comment_paths, resolved_submission_paths),
        "fingerprint": fingerprint_hash(_source_payload(resolved_comment_paths, resolved_submission_paths)),
        "n_rows": int(len(monthly)),
    }
    table_paths["metadata"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return AiMentionsArtifacts(monthly=monthly, table_paths=table_paths, figure_paths=figure_paths, metadata=payload)


def load_ai_mentions_cache(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return cache metadata when AI-mention outputs are current."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    table_path = out_tables / AI_MENTIONS_FILENAME
    metadata_path = out_tables / AI_MENTIONS_METADATA_FILENAME
    figure_paths = [
        out_figures / "ai-mentions-health-vs-general-comments.svg",
        out_figures / "ai-mentions-health-vs-general-submissions.svg",
        out_figures / "ai-mentions-top-subreddits.svg",
    ]
    if not table_path.exists() or not metadata_path.exists() or not all(path.exists() for path in figure_paths):
        return None

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_sources = _source_payload(resolved_comment_paths, resolved_submission_paths)
    if payload.get("fingerprint") != fingerprint_hash(current_sources):
        return None
    return payload


def _source_payload(comment_paths: list[Path], submission_paths: list[Path]) -> dict[str, Any]:
    return {
        "version": AI_MENTIONS_CACHE_VERSION,
        "comment_files": fingerprint_paths(comment_paths),
        "submission_files": fingerprint_paths(submission_paths),
    }


def _count_ai_mentions(paths: list[Path], *, field: str) -> list[tuple[str, str, int]]:
    counts: dict[tuple[str, str], int] = {}
    for path in paths:
        log.info("Scanning AI mentions in %s …", path.name)
        for record in stream_zst(path):
            created_utc = extract_created_utc(record)
            if created_utc is None:
                continue
            subreddit = str(record.get("subreddit", "unknown"))
            month = epoch_to_month(created_utc)
            if field == "submission":
                text = f"{record.get('title', '')} {record.get('selftext', '')}"
            else:
                text = str(record.get(field, ""))
            if AI_MENTION_PATTERN.search(text):
                key = (subreddit, month)
                counts[key] = counts.get(key, 0) + 1
    return [(subreddit, month, count) for (subreddit, month), count in sorted(counts.items())]


def _plot_type_trends(frame: pd.DataFrame, column: str, out_path: Path, title: str) -> None:
    plot_frame = (
        frame.loc[frame["community_type"].isin(["health", "general"])]
        .groupby(["month", "community_type"], as_index=False)[column]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
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
    ax.set_xlabel("Month")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_top_subreddits(frame: pd.DataFrame, out_path: Path) -> None:
    top = (
        frame.assign(total_mentions=frame["ai_mention_comments"] + frame["ai_mention_submissions"])
        .groupby("subreddit", as_index=False)["total_mentions"]
        .sum()
        .sort_values("total_mentions", ascending=False, kind="stable")
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(top["subreddit"], top["total_mentions"], color="#1f78b4")
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("AI mentions")
    ax.set_title("Top subreddits by AI mentions")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
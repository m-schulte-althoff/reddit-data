"""Community responsiveness and support-availability metrics."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, OUTPUT_DIR, TABLES_DIR
from src.helpers import classify_subreddit
from src.io_utils import (
    discover_filtered_paths,
    epoch_to_month,
    extract_created_utc,
    fingerprint_hash,
    fingerprint_paths,
    is_deleted_removed,
    iter_month_range,
    stream_zst,
)

log = logging.getLogger(__name__)

RESPONSIVENESS_POSTS_FILENAME = "responsiveness-posts.csv"
RESPONSIVENESS_MONTHLY_FILENAME = "responsiveness-monthly.csv"
RESPONSIVENESS_METADATA_FILENAME = "responsiveness-metadata.json"
RESPONSIVENESS_CACHE_VERSION = 1


@dataclass(frozen=True)
class SubmissionMeta:
    """Minimal submission metadata needed for responsiveness metrics."""

    submission_id: str
    subreddit: str
    month: str
    author: str
    created_utc: int


@dataclass(frozen=True)
class PendingDepthComment:
    """Comment waiting for its parent depth to resolve."""

    comment_id: str
    submission_id: str


@dataclass
class ResponsivenessArtifacts:
    """Written outputs plus in-memory tables."""

    posts: pd.DataFrame
    monthly: pd.DataFrame
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    metadata: dict[str, Any]


def run_responsiveness_analysis(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> ResponsivenessArtifacts:
    """Compute, cache, and write responsiveness outputs."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_cache = cache_dir or (OUTPUT_DIR / "cache")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_cache.mkdir(parents=True, exist_ok=True)

    metadata = load_responsiveness_cache(
        resolved_comment_paths,
        resolved_submission_paths,
        tables_dir=out_tables,
        figures_dir=out_figures,
    )
    table_paths = _table_paths(out_tables)
    figure_paths = _figure_paths(out_figures)
    if metadata is not None:
        log.info("Using cached responsiveness outputs at %s", table_paths["posts"])
        return ResponsivenessArtifacts(
            posts=pd.read_csv(table_paths["posts"]),
            monthly=pd.read_csv(table_paths["monthly"]),
            table_paths=table_paths,
            figure_paths=figure_paths,
            metadata=metadata,
        )

    sqlite_path = out_cache / "responsiveness.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()

    submissions = _load_submissions_into_sqlite(sqlite_path, resolved_submission_paths)
    _stream_comments_into_sqlite(sqlite_path, submissions, resolved_comment_paths)
    posts = _load_post_level_frame(sqlite_path)
    monthly = _build_monthly_frame(posts)
    _write_outputs(posts, monthly, table_paths)
    _plot_outputs(posts, figure_paths)

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "fingerprint": fingerprint_hash(_source_payload(resolved_comment_paths, resolved_submission_paths)),
        "sources": _source_payload(resolved_comment_paths, resolved_submission_paths),
        "n_posts": int(len(posts)),
        "n_monthly_rows": int(len(monthly)),
        "sqlite_path": str(sqlite_path),
    }
    table_paths["metadata"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ResponsivenessArtifacts(
        posts=posts,
        monthly=monthly,
        table_paths=table_paths,
        figure_paths=figure_paths,
        metadata=payload,
    )


def load_responsiveness_cache(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return cache metadata when responsiveness outputs are current."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    table_paths = _table_paths(out_tables)
    figure_paths = _figure_paths(out_figures)

    required_paths = list(table_paths.values()) + list(figure_paths.values())
    if not all(path.exists() for path in required_paths):
        return None

    try:
        payload = json.loads(table_paths["metadata"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read responsiveness metadata: %s", exc)
        return None

    current_sources = _source_payload(resolved_comment_paths, resolved_submission_paths)
    if payload.get("fingerprint") != fingerprint_hash(current_sources):
        log.info("Responsiveness cache stale (fingerprint mismatch) — will recompute")
        return None
    return payload


def _table_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "posts": base_dir / RESPONSIVENESS_POSTS_FILENAME,
        "monthly": base_dir / RESPONSIVENESS_MONTHLY_FILENAME,
        "metadata": base_dir / RESPONSIVENESS_METADATA_FILENAME,
    }


def _figure_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "reply_rate": base_dir / "responsiveness-reply-rate-health-vs-general.svg",
        "latency": base_dir / "responsiveness-latency-health-vs-general.svg",
        "op_followup": base_dir / "responsiveness-op-followup-health-vs-general.svg",
        "unanswered_rate": base_dir / "responsiveness-unanswered-rate-health-vs-general.svg",
    }


def _source_payload(comment_paths: list[Path], submission_paths: list[Path]) -> dict[str, Any]:
    return {
        "version": RESPONSIVENESS_CACHE_VERSION,
        "comment_files": fingerprint_paths(comment_paths),
        "submission_files": fingerprint_paths(submission_paths),
    }


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE submissions (
            submission_id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            created_utc INTEGER NOT NULL
        );

        CREATE TABLE post_metrics (
            submission_id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            created_utc INTEGER NOT NULL,
            num_comments_observed INTEGER NOT NULL DEFAULT 0,
            first_reply_latency_minutes REAL,
            op_followup_comments INTEGER NOT NULL DEFAULT 0,
            direct_reply_count INTEGER NOT NULL DEFAULT 0,
            deep_reply_count INTEGER NOT NULL DEFAULT 0,
            max_depth INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE commenter_counts (
            submission_id TEXT NOT NULL,
            commenter TEXT NOT NULL,
            is_non_op INTEGER NOT NULL,
            comment_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (submission_id, commenter)
        );
        """,
    )


def _load_submissions_into_sqlite(
    sqlite_path: Path,
    submission_paths: list[Path],
) -> dict[str, SubmissionMeta]:
    submissions: dict[str, SubmissionMeta] = {}
    conn = _connect(sqlite_path)
    _init_schema(conn)

    submission_rows: list[tuple[str, str, str, str, int]] = []
    post_rows: list[tuple[str, str, str, str, int]] = []

    for path in submission_paths:
        log.info("Loading submission metadata from %s …", path.name)
        for record in stream_zst(path):
            submission_id = str(record.get("id", ""))
            created_utc = extract_created_utc(record)
            if not submission_id or created_utc is None:
                continue

            subreddit = str(record.get("subreddit", "unknown"))
            month = epoch_to_month(created_utc)
            author = str(record.get("author", ""))
            meta = SubmissionMeta(
                submission_id=submission_id,
                subreddit=subreddit,
                month=month,
                author=author,
                created_utc=created_utc,
            )
            submissions[submission_id] = meta
            submission_rows.append((submission_id, subreddit, month, author, created_utc))
            post_rows.append((submission_id, subreddit, month, author, created_utc))

    with conn:
        conn.executemany(
            "INSERT INTO submissions (submission_id, subreddit, month, author, created_utc) VALUES (?, ?, ?, ?, ?)",
            submission_rows,
        )
        conn.executemany(
            "INSERT INTO post_metrics (submission_id, subreddit, month, author, created_utc) VALUES (?, ?, ?, ?, ?)",
            post_rows,
        )
    conn.close()
    return submissions


def _stream_comments_into_sqlite(
    sqlite_path: Path,
    submissions: dict[str, SubmissionMeta],
    comment_paths: list[Path],
) -> None:
    conn = _connect(sqlite_path)
    depth_map: dict[str, int] = {}
    pending: dict[str, list[PendingDepthComment]] = {}

    with conn:
        for path in comment_paths:
            log.info("Updating responsiveness metrics from %s …", path.name)
            for record in stream_zst(path):
                comment_id = str(record.get("id", ""))
                if not comment_id:
                    continue

                link_id = str(record.get("link_id", ""))
                if not link_id.startswith("t3_"):
                    continue
                submission_id = link_id[3:]
                submission_meta = submissions.get(submission_id)
                if submission_meta is None:
                    continue

                created_utc = extract_created_utc(record)
                if created_utc is None:
                    continue

                author = str(record.get("author", ""))
                _record_comment_activity(conn, submission_meta, author, created_utc)

                parent_id = str(record.get("parent_id", ""))
                if parent_id.startswith("t3_"):
                    depth = 1
                elif parent_id.startswith("t1_"):
                    parent_key = parent_id[3:]
                    if parent_key in depth_map:
                        depth = depth_map[parent_key] + 1
                    else:
                        pending.setdefault(parent_key, []).append(
                            PendingDepthComment(comment_id=comment_id, submission_id=submission_id),
                        )
                        continue
                else:
                    continue

                _record_comment_depth(conn, submission_id, comment_id, depth, depth_map, pending)

    conn.close()


def _record_comment_activity(
    conn: sqlite3.Connection,
    submission_meta: SubmissionMeta,
    author: str,
    created_utc: int,
) -> None:
    latency_minutes = max(0.0, (created_utc - submission_meta.created_utc) / 60.0)
    is_op = 1 if author == submission_meta.author else 0

    conn.execute(
        """
        UPDATE post_metrics
        SET num_comments_observed = num_comments_observed + 1,
            first_reply_latency_minutes = CASE
                WHEN first_reply_latency_minutes IS NULL OR ? < first_reply_latency_minutes THEN ?
                ELSE first_reply_latency_minutes
            END,
            op_followup_comments = op_followup_comments + ?
        WHERE submission_id = ?
        """,
        (latency_minutes, latency_minutes, is_op, submission_meta.submission_id),
    )

    if author and not is_deleted_removed(author):
        is_non_op = 0 if author == submission_meta.author else 1
        conn.execute(
            """
            INSERT INTO commenter_counts (submission_id, commenter, is_non_op, comment_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(submission_id, commenter) DO UPDATE SET
                comment_count = commenter_counts.comment_count + 1,
                is_non_op = MAX(commenter_counts.is_non_op, excluded.is_non_op)
            """,
            (submission_meta.submission_id, author, is_non_op),
        )


def _record_comment_depth(
    conn: sqlite3.Connection,
    submission_id: str,
    comment_id: str,
    depth: int,
    depth_map: dict[str, int],
    pending: dict[str, list[PendingDepthComment]],
) -> None:
    depth_map[comment_id] = depth
    _update_depth_metrics(conn, submission_id, depth)

    stack: list[tuple[str, str, int]] = [(comment_id, submission_id, depth)]
    while stack:
        parent_comment_id, parent_submission_id, parent_depth = stack.pop()
        for child in pending.pop(parent_comment_id, []):
            child_depth = parent_depth + 1
            depth_map[child.comment_id] = child_depth
            _update_depth_metrics(conn, child.submission_id, child_depth)
            stack.append((child.comment_id, child.submission_id, child_depth))


def _update_depth_metrics(conn: sqlite3.Connection, submission_id: str, depth: int) -> None:
    conn.execute(
        """
        UPDATE post_metrics
        SET direct_reply_count = direct_reply_count + ?,
            deep_reply_count = deep_reply_count + ?,
            max_depth = CASE WHEN ? > max_depth THEN ? ELSE max_depth END
        WHERE submission_id = ?
        """,
        (
            1 if depth == 1 else 0,
            1 if depth >= 2 else 0,
            depth,
            depth,
            submission_id,
        ),
    )


def _load_post_level_frame(sqlite_path: Path) -> pd.DataFrame:
    conn = _connect(sqlite_path)
    query = """
        SELECT
            p.submission_id,
            p.subreddit,
            p.month,
            p.author,
            p.created_utc,
            p.num_comments_observed,
            CASE WHEN p.num_comments_observed > 0 THEN 1 ELSE 0 END AS has_reply,
            p.first_reply_latency_minutes,
            CASE
                WHEN p.first_reply_latency_minutes IS NULL THEN 0.0
                ELSE p.first_reply_latency_minutes / 60.0
            END AS first_reply_latency_hours,
            COALESCE(c.unique_commenters, 0) AS unique_commenters,
            COALESCE(c.unique_non_op_commenters, 0) AS unique_non_op_commenters,
            p.op_followup_comments,
            CASE
                WHEN p.num_comments_observed = 0 THEN 0.0
                ELSE CAST(p.op_followup_comments AS REAL) / p.num_comments_observed
            END AS op_followup_share,
            p.direct_reply_count,
            p.deep_reply_count,
            p.max_depth,
            CASE
                WHEN p.num_comments_observed = 0 THEN 0.0
                ELSE CAST(p.deep_reply_count AS REAL) / p.num_comments_observed
            END AS threading_ratio,
            COALESCE(c.top_helper_comment_count_on_post, 0) AS top_helper_comment_count_on_post
        FROM post_metrics p
        LEFT JOIN (
            SELECT
                submission_id,
                COUNT(*) AS unique_commenters,
                SUM(CASE WHEN is_non_op = 1 THEN 1 ELSE 0 END) AS unique_non_op_commenters,
                COALESCE(MAX(CASE WHEN is_non_op = 1 THEN comment_count ELSE 0 END), 0) AS top_helper_comment_count_on_post
            FROM commenter_counts
            GROUP BY submission_id
        ) c
        ON c.submission_id = p.submission_id
        ORDER BY p.subreddit, p.month, p.submission_id
    """
    frame = pd.read_sql_query(query, conn)
    conn.close()
    return frame


def _build_monthly_frame(posts: pd.DataFrame) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(columns=[
            "subreddit",
            "month",
            "community_type",
            "submissions",
            "reply_rate",
            "unanswered_rate",
            "median_first_reply_latency_hours",
            "p25_first_reply_latency_hours",
            "p75_first_reply_latency_hours",
            "mean_unique_commenters",
            "median_unique_commenters",
            "mean_non_op_commenters",
            "op_followup_rate",
            "mean_op_followup_comments",
            "mean_post_threading_ratio",
            "mean_post_max_depth",
        ])

    rows: list[dict[str, Any]] = []
    grouped = posts.groupby(["subreddit", "month"], sort=True)
    for (subreddit, month), group in grouped:
        replied = group.loc[group["has_reply"] == 1, "first_reply_latency_hours"]
        rows.append({
            "subreddit": subreddit,
            "month": month,
            "community_type": classify_subreddit(subreddit),
            "submissions": int(len(group)),
            "reply_rate": float(group["has_reply"].mean()),
            "unanswered_rate": float(1.0 - group["has_reply"].mean()),
            "median_first_reply_latency_hours": _series_stat(replied, "median"),
            "p25_first_reply_latency_hours": _series_stat(replied, 0.25),
            "p75_first_reply_latency_hours": _series_stat(replied, 0.75),
            "mean_unique_commenters": float(group["unique_commenters"].mean()),
            "median_unique_commenters": float(group["unique_commenters"].median()),
            "mean_non_op_commenters": float(group["unique_non_op_commenters"].mean()),
            "op_followup_rate": float((group["op_followup_comments"] > 0).mean()),
            "mean_op_followup_comments": float(group["op_followup_comments"].mean()),
            "mean_post_threading_ratio": float(group["threading_ratio"].mean()),
            "mean_post_max_depth": float(group["max_depth"].mean()),
        })

    monthly = pd.DataFrame(rows)
    all_subreddits = sorted(posts["subreddit"].unique().tolist(), key=str.casefold)
    all_months = iter_month_range(posts["month"].min(), posts["month"].max())
    full_rows: list[dict[str, Any]] = []
    monthly_by_key = {
        (row["subreddit"], row["month"]): row
        for row in monthly.to_dict(orient="records")
    }

    for subreddit in all_subreddits:
        for month in all_months:
            row = monthly_by_key.get((subreddit, month))
            if row is None:
                full_rows.append({
                    "subreddit": subreddit,
                    "month": month,
                    "community_type": classify_subreddit(subreddit),
                    "submissions": 0,
                    "reply_rate": 0.0,
                    "unanswered_rate": 0.0,
                    "median_first_reply_latency_hours": 0.0,
                    "p25_first_reply_latency_hours": 0.0,
                    "p75_first_reply_latency_hours": 0.0,
                    "mean_unique_commenters": 0.0,
                    "median_unique_commenters": 0.0,
                    "mean_non_op_commenters": 0.0,
                    "op_followup_rate": 0.0,
                    "mean_op_followup_comments": 0.0,
                    "mean_post_threading_ratio": 0.0,
                    "mean_post_max_depth": 0.0,
                })
            else:
                full_rows.append(row)

    return pd.DataFrame(full_rows).sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)


def _series_stat(series: pd.Series, stat: float | str) -> float:
    if series.empty:
        return 0.0
    if stat == "median":
        value = float(series.median())
    else:
        value = float(series.quantile(stat))
    return 0.0 if pd.isna(value) else value


def _write_outputs(posts: pd.DataFrame, monthly: pd.DataFrame, table_paths: dict[str, Path]) -> None:
    posts.to_csv(table_paths["posts"], index=False)
    monthly.to_csv(table_paths["monthly"], index=False)
    log.info("Wrote %s", table_paths["posts"])
    log.info("Wrote %s", table_paths["monthly"])


def _plot_outputs(posts: pd.DataFrame, figure_paths: dict[str, Path]) -> None:
    plot_frame = posts.copy()
    plot_frame["community_type"] = plot_frame["subreddit"].map(classify_subreddit)
    plot_frame = plot_frame.loc[plot_frame["community_type"].isin(["health", "general"])]

    if plot_frame.empty:
        for path in figure_paths.values():
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No classified health/general posts available.", ha="center", va="center")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
        return

    monthly = (
        plot_frame.groupby(["month", "community_type"], as_index=False)
        .agg(
            reply_rate=("has_reply", "mean"),
            unanswered_rate=("has_reply", lambda s: 1.0 - float(s.mean())),
            median_first_reply_latency_hours=(
                "first_reply_latency_hours",
                lambda s: float(s.loc[s > 0].median()) if (s > 0).any() else 0.0,
            ),
            op_followup_rate=("op_followup_comments", lambda s: float((s > 0).mean())),
        )
    )

    _plot_type_trend(monthly, "reply_rate", "Reply rate", figure_paths["reply_rate"])
    _plot_type_trend(
        monthly,
        "median_first_reply_latency_hours",
        "Median first-reply latency (hours)",
        figure_paths["latency"],
    )
    _plot_type_trend(monthly, "op_followup_rate", "OP follow-up rate", figure_paths["op_followup"])
    _plot_type_trend(monthly, "unanswered_rate", "Unanswered rate", figure_paths["unanswered_rate"])


def _plot_type_trend(frame: pd.DataFrame, value_col: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    styles = {
        "health": {"label": "Health", "color": "#1b9e77", "marker": "o"},
        "general": {"label": "General", "color": "#d95f02", "marker": "s"},
    }
    for community_type in ["health", "general"]:
        subset = frame.loc[frame["community_type"] == community_type]
        if subset.empty:
            continue
        style = styles[community_type]
        ax.plot(
            subset["month"],
            subset[value_col],
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            label=style["label"],
        )

    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
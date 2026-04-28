"""Bond-vs-identity interaction metrics for subreddit-month panels."""

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

INTERACTIONS_MONTHLY_FILENAME = "interactions-monthly.csv"
INTERACTIONS_METADATA_FILENAME = "interactions-metadata.json"
INTERACTIONS_CACHE_VERSION = 1


@dataclass(frozen=True)
class SubmissionMeta:
    """Minimal submission metadata needed for thread interaction metrics."""

    submission_id: str
    subreddit: str
    month: str
    author: str
    created_utc: int


@dataclass(frozen=True)
class PendingComment:
    """Comment waiting for its parent comment to resolve."""

    comment_id: str
    submission_id: str
    subreddit: str
    edge_month: str
    author: str


@dataclass
class InteractionsArtifacts:
    """Written outputs plus the monthly interactions table."""

    monthly: pd.DataFrame
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    metadata: dict[str, Any]


def run_interactions_analysis(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> InteractionsArtifacts:
    """Compute monthly bond-vs-identity interaction metrics."""
    resolved_comment_paths = comment_paths or discover_filtered_paths("comments")
    resolved_submission_paths = submission_paths or discover_filtered_paths("submissions")
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_cache = cache_dir or (OUTPUT_DIR / "cache")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_cache.mkdir(parents=True, exist_ok=True)

    table_paths = _table_paths(out_tables)
    figure_paths = _figure_paths(out_figures)
    metadata = load_interactions_cache(
        resolved_comment_paths,
        resolved_submission_paths,
        tables_dir=out_tables,
        figures_dir=out_figures,
    )
    if metadata is not None:
        log.info("Using cached interactions output at %s", table_paths["monthly"])
        return InteractionsArtifacts(
            monthly=pd.read_csv(table_paths["monthly"]),
            table_paths=table_paths,
            figure_paths=figure_paths,
            metadata=metadata,
        )

    sqlite_path = out_cache / "interactions.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()

    submissions = _load_submissions_into_sqlite(sqlite_path, resolved_submission_paths)
    _stream_comments_into_sqlite(sqlite_path, submissions, resolved_comment_paths)
    monthly = _build_monthly_frame(sqlite_path)
    monthly.to_csv(table_paths["monthly"], index=False)
    _plot_outputs(monthly, figure_paths)

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "fingerprint": fingerprint_hash(_source_payload(resolved_comment_paths, resolved_submission_paths)),
        "sources": _source_payload(resolved_comment_paths, resolved_submission_paths),
        "sqlite_path": str(sqlite_path),
        "n_rows": int(len(monthly)),
    }
    table_paths["metadata"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return InteractionsArtifacts(
        monthly=monthly,
        table_paths=table_paths,
        figure_paths=figure_paths,
        metadata=payload,
    )


def load_interactions_cache(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return cache metadata when monthly interaction outputs are current."""
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
        log.warning("Could not read interactions metadata: %s", exc)
        return None

    current_sources = _source_payload(resolved_comment_paths, resolved_submission_paths)
    if payload.get("fingerprint") != fingerprint_hash(current_sources):
        log.info("Interactions cache stale (fingerprint mismatch) — will recompute")
        return None
    return payload


def _table_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "monthly": base_dir / INTERACTIONS_MONTHLY_FILENAME,
        "metadata": base_dir / INTERACTIONS_METADATA_FILENAME,
    }


def _figure_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "bond": base_dir / "interactions-bond-index-health-vs-general.svg",
        "identity": base_dir / "interactions-identity-index-health-vs-general.svg",
    }


def _source_payload(comment_paths: list[Path], submission_paths: list[Path]) -> dict[str, Any]:
    return {
        "version": INTERACTIONS_CACHE_VERSION,
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
            num_comments_observed INTEGER NOT NULL DEFAULT 0,
            op_returned INTEGER NOT NULL DEFAULT 0,
            deep_reply_count INTEGER NOT NULL DEFAULT 0,
            max_depth INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE helper_counts (
            submission_id TEXT NOT NULL,
            commenter TEXT NOT NULL,
            is_non_op INTEGER NOT NULL,
            comment_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (submission_id, commenter)
        );

        CREATE TABLE author_activity (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            activity_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (subreddit, month, author)
        );

        CREATE TABLE reply_edges (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            source_author TEXT NOT NULL,
            target_author TEXT NOT NULL,
            edge_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (subreddit, month, source_author, target_author)
        );
        """,
    )


def _valid_author(author: str) -> str | None:
    normalized = author.strip()
    if not normalized or is_deleted_removed(normalized):
        return None
    return normalized


def _load_submissions_into_sqlite(
    sqlite_path: Path,
    submission_paths: list[Path],
) -> dict[str, SubmissionMeta]:
    submissions: dict[str, SubmissionMeta] = {}
    conn = _connect(sqlite_path)
    _init_schema(conn)

    submission_rows: list[tuple[str, str, str, str, int]] = []
    post_rows: list[tuple[str, str, str, str]] = []

    with conn:
        for path in submission_paths:
            log.info("Loading interaction submission metadata from %s …", path.name)
            for record in stream_zst(path):
                submission_id = str(record.get("id", ""))
                created_utc = extract_created_utc(record)
                if not submission_id or created_utc is None:
                    continue

                subreddit = str(record.get("subreddit", "unknown"))
                month = epoch_to_month(created_utc)
                author = str(record.get("author", ""))
                submissions[submission_id] = SubmissionMeta(
                    submission_id=submission_id,
                    subreddit=subreddit,
                    month=month,
                    author=author,
                    created_utc=created_utc,
                )
                submission_rows.append((submission_id, subreddit, month, author, created_utc))
                post_rows.append((submission_id, subreddit, month, author))
                valid_author = _valid_author(author)
                if valid_author is not None:
                    _record_author_activity(conn, subreddit, month, valid_author)

        conn.executemany(
            "INSERT INTO submissions (submission_id, subreddit, month, author, created_utc) VALUES (?, ?, ?, ?, ?)",
            submission_rows,
        )
        conn.executemany(
            "INSERT INTO post_metrics (submission_id, subreddit, month, author) VALUES (?, ?, ?, ?)",
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
    comment_state: dict[str, tuple[str, str, int]] = {}
    pending: dict[str, list[PendingComment]] = {}

    with conn:
        for path in comment_paths:
            log.info("Updating interaction metrics from %s …", path.name)
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

                subreddit = submission_meta.subreddit
                edge_month = epoch_to_month(created_utc)
                author = str(record.get("author", ""))
                valid_author = _valid_author(author)
                if valid_author is not None:
                    _record_author_activity(conn, subreddit, edge_month, valid_author)
                _record_post_comment(
                    conn,
                    submission_id,
                    is_op=1 if author == submission_meta.author else 0,
                )
                _record_helper_count(conn, submission_id, author, submission_meta.author)

                parent_id = str(record.get("parent_id", ""))
                if parent_id.startswith("t3_"):
                    depth = 1
                    parent_author = submission_meta.author
                elif parent_id.startswith("t1_"):
                    parent_comment_id = parent_id[3:]
                    if parent_comment_id in comment_state:
                        _, parent_author, parent_depth = comment_state[parent_comment_id]
                        depth = parent_depth + 1
                    else:
                        pending.setdefault(parent_comment_id, []).append(
                            PendingComment(
                                comment_id=comment_id,
                                submission_id=submission_id,
                                subreddit=subreddit,
                                edge_month=edge_month,
                                author=author,
                            ),
                        )
                        continue
                else:
                    continue

                _resolve_comment(
                    conn,
                    comment_id,
                    submission_id,
                    subreddit,
                    edge_month,
                    author,
                    parent_author,
                    depth,
                    comment_state,
                    pending,
                )

    conn.close()


def _record_author_activity(conn: sqlite3.Connection, subreddit: str, month: str, author: str) -> None:
    conn.execute(
        """
        INSERT INTO author_activity (subreddit, month, author, activity_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(subreddit, month, author) DO UPDATE SET
            activity_count = author_activity.activity_count + 1
        """,
        (subreddit, month, author),
    )


def _record_post_comment(conn: sqlite3.Connection, submission_id: str, *, is_op: int) -> None:
    conn.execute(
        """
        UPDATE post_metrics
        SET num_comments_observed = num_comments_observed + 1,
            op_returned = CASE WHEN ? = 1 THEN 1 ELSE op_returned END
        WHERE submission_id = ?
        """,
        (is_op, submission_id),
    )


def _record_helper_count(
    conn: sqlite3.Connection,
    submission_id: str,
    author: str,
    submission_author: str,
) -> None:
    valid_author = _valid_author(author)
    if valid_author is None:
        return

    is_non_op = 0 if author == submission_author else 1
    conn.execute(
        """
        INSERT INTO helper_counts (submission_id, commenter, is_non_op, comment_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(submission_id, commenter) DO UPDATE SET
            comment_count = helper_counts.comment_count + 1,
            is_non_op = MAX(helper_counts.is_non_op, excluded.is_non_op)
        """,
        (submission_id, valid_author, is_non_op),
    )


def _resolve_comment(
    conn: sqlite3.Connection,
    comment_id: str,
    submission_id: str,
    subreddit: str,
    edge_month: str,
    author: str,
    parent_author: str,
    depth: int,
    comment_state: dict[str, tuple[str, str, int]],
    pending: dict[str, list[PendingComment]],
) -> None:
    comment_state[comment_id] = (submission_id, author, depth)
    _record_depth_and_edge(conn, submission_id, subreddit, edge_month, author, parent_author, depth)

    stack: list[tuple[str, str, str, str, str, int]] = [
        (comment_id, submission_id, subreddit, edge_month, author, depth),
    ]
    while stack:
        parent_comment_id, parent_submission_id, parent_subreddit, _, resolved_author, resolved_depth = stack.pop()
        for child in pending.pop(parent_comment_id, []):
            child_depth = resolved_depth + 1
            comment_state[child.comment_id] = (child.submission_id, child.author, child_depth)
            _record_depth_and_edge(
                conn,
                child.submission_id,
                child.subreddit,
                child.edge_month,
                child.author,
                resolved_author,
                child_depth,
            )
            stack.append(
                (
                    child.comment_id,
                    child.submission_id,
                    parent_subreddit,
                    child.edge_month,
                    child.author,
                    child_depth,
                ),
            )


def _record_depth_and_edge(
    conn: sqlite3.Connection,
    submission_id: str,
    subreddit: str,
    edge_month: str,
    author: str,
    parent_author: str,
    depth: int,
) -> None:
    conn.execute(
        """
        UPDATE post_metrics
        SET deep_reply_count = deep_reply_count + ?,
            max_depth = CASE WHEN ? > max_depth THEN ? ELSE max_depth END
        WHERE submission_id = ?
        """,
        (1 if depth >= 2 else 0, depth, depth, submission_id),
    )

    source_author = _valid_author(author)
    target_author = _valid_author(parent_author)
    if source_author is None or target_author is None or source_author == target_author:
        return

    conn.execute(
        """
        INSERT INTO reply_edges (subreddit, month, source_author, target_author, edge_count)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(subreddit, month, source_author, target_author) DO UPDATE SET
            edge_count = reply_edges.edge_count + 1
        """,
        (subreddit, edge_month, source_author, target_author),
    )


def _build_monthly_frame(sqlite_path: Path) -> pd.DataFrame:
    conn = _connect(sqlite_path)
    post_frame = pd.read_sql_query(
        """
        SELECT
            p.submission_id,
            p.subreddit,
            p.month,
            p.num_comments_observed,
            p.op_returned,
            p.max_depth,
            COALESCE(h.unique_non_op_commenters, 0) AS unique_non_op_commenters,
            COALESCE(h.non_op_comment_total, 0) AS non_op_comment_total,
            COALESCE(h.max_non_op_comment_count, 0) AS max_non_op_comment_count
        FROM post_metrics p
        LEFT JOIN (
            SELECT
                submission_id,
                SUM(CASE WHEN is_non_op = 1 THEN 1 ELSE 0 END) AS unique_non_op_commenters,
                SUM(CASE WHEN is_non_op = 1 THEN comment_count ELSE 0 END) AS non_op_comment_total,
                COALESCE(MAX(CASE WHEN is_non_op = 1 THEN comment_count ELSE 0 END), 0) AS max_non_op_comment_count
            FROM helper_counts
            GROUP BY submission_id
        ) h
        ON h.submission_id = p.submission_id
        ORDER BY p.subreddit, p.month, p.submission_id
        """,
        conn,
    )
    author_activity = pd.read_sql_query(
        "SELECT subreddit, month, author FROM author_activity ORDER BY subreddit, month, author",
        conn,
    )
    reply_edges = pd.read_sql_query(
        "SELECT subreddit, month, source_author, target_author, edge_count FROM reply_edges ORDER BY subreddit, month, source_author, target_author",
        conn,
    )
    conn.close()

    return _assemble_monthly_frame(post_frame, author_activity, reply_edges)


def _assemble_monthly_frame(
    post_frame: pd.DataFrame,
    author_activity: pd.DataFrame,
    reply_edges: pd.DataFrame,
) -> pd.DataFrame:
    months = _all_months(post_frame, author_activity, reply_edges)
    subreddits = _all_subreddits(post_frame, author_activity, reply_edges)
    if not months or not subreddits:
        return pd.DataFrame(columns=[
            "subreddit",
            "month",
            "community_type",
            "unique_authors",
            "new_author_share",
            "returning_author_share",
            "repeat_author_share",
            "op_return_rate",
            "reciprocal_dyad_share",
            "repeat_dyad_share",
            "single_commenter_thread_share",
            "multi_actor_thread_share",
            "focused_thread_share",
            "distributed_thread_share",
            "bond_index",
            "identity_index",
        ])

    author_metrics = _author_history_metrics(author_activity, subreddits, months)
    post_metrics = _post_monthly_metrics(post_frame)
    dyad_metrics = _dyad_monthly_metrics(reply_edges)

    rows: list[dict[str, Any]] = []
    for subreddit in subreddits:
        for month in months:
            author_row = author_metrics.get((subreddit, month), {})
            post_row = post_metrics.get((subreddit, month), {})
            dyad_row = dyad_metrics.get((subreddit, month), {})
            rows.append({
                "subreddit": subreddit,
                "month": month,
                "community_type": classify_subreddit(subreddit),
                "unique_authors": int(author_row.get("unique_authors", 0)),
                "new_author_share": float(author_row.get("new_author_share", 0.0)),
                "returning_author_share": float(author_row.get("returning_author_share", 0.0)),
                "repeat_author_share": float(author_row.get("repeat_author_share", 0.0)),
                "op_return_rate": float(post_row.get("op_return_rate", 0.0)),
                "reciprocal_dyad_share": float(dyad_row.get("reciprocal_dyad_share", 0.0)),
                "repeat_dyad_share": float(dyad_row.get("repeat_dyad_share", 0.0)),
                "single_commenter_thread_share": float(post_row.get("single_commenter_thread_share", 0.0)),
                "multi_actor_thread_share": float(post_row.get("multi_actor_thread_share", 0.0)),
                "focused_thread_share": float(post_row.get("focused_thread_share", 0.0)),
                "distributed_thread_share": float(post_row.get("distributed_thread_share", 0.0)),
            })

    monthly = pd.DataFrame(rows).sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)
    active_mask = monthly["unique_authors"] > 0
    monthly["bond_index"] = (
        _zscore(monthly["op_return_rate"], active_mask)
        + _zscore(monthly["reciprocal_dyad_share"], active_mask)
        + _zscore(monthly["repeat_dyad_share"], active_mask)
        + _zscore(monthly["repeat_author_share"], active_mask)
    )
    monthly["identity_index"] = (
        _zscore(monthly["unique_authors"].astype(float), active_mask)
        + _zscore(monthly["new_author_share"], active_mask)
        + _zscore(monthly["distributed_thread_share"], active_mask)
        - _zscore(monthly["focused_thread_share"], active_mask)
    )
    monthly.loc[~active_mask, ["bond_index", "identity_index"]] = 0.0
    return monthly


def _all_months(post_frame: pd.DataFrame, author_activity: pd.DataFrame, reply_edges: pd.DataFrame) -> list[str]:
    month_values = pd.concat([
        post_frame.get("month", pd.Series(dtype=str)),
        author_activity.get("month", pd.Series(dtype=str)),
        reply_edges.get("month", pd.Series(dtype=str)),
    ], ignore_index=True).dropna()
    if month_values.empty:
        return []
    unique_months = sorted(month_values.astype(str).unique().tolist())
    return iter_month_range(unique_months[0], unique_months[-1])


def _all_subreddits(post_frame: pd.DataFrame, author_activity: pd.DataFrame, reply_edges: pd.DataFrame) -> list[str]:
    subreddit_values = pd.concat([
        post_frame.get("subreddit", pd.Series(dtype=str)),
        author_activity.get("subreddit", pd.Series(dtype=str)),
        reply_edges.get("subreddit", pd.Series(dtype=str)),
    ], ignore_index=True).dropna()
    return sorted(subreddit_values.astype(str).unique().tolist(), key=str.casefold)


def _author_history_metrics(
    author_activity: pd.DataFrame,
    subreddits: list[str],
    months: list[str],
) -> dict[tuple[str, str], dict[str, float | int]]:
    author_sets: dict[tuple[str, str], set[str]] = {}
    for (subreddit, month), group in author_activity.groupby(["subreddit", "month"], sort=True):
        author_sets[(str(subreddit), str(month))] = set(group["author"].astype(str))

    metrics: dict[tuple[str, str], dict[str, float | int]] = {}
    for subreddit in subreddits:
        seen_before: set[str] = set()
        previous_month_authors: set[str] = set()
        for month in months:
            current_authors = author_sets.get((subreddit, month), set())
            unique_authors = len(current_authors)
            if unique_authors == 0:
                metrics[(subreddit, month)] = {
                    "unique_authors": 0,
                    "new_author_share": 0.0,
                    "returning_author_share": 0.0,
                    "repeat_author_share": 0.0,
                }
                previous_month_authors = set()
                continue

            repeat_authors = current_authors & previous_month_authors
            new_authors = current_authors - seen_before
            returning_authors = (current_authors & seen_before) - previous_month_authors
            metrics[(subreddit, month)] = {
                "unique_authors": unique_authors,
                "new_author_share": len(new_authors) / unique_authors,
                "returning_author_share": len(returning_authors) / unique_authors,
                "repeat_author_share": len(repeat_authors) / unique_authors,
            }
            seen_before |= current_authors
            previous_month_authors = current_authors

    return metrics


def _post_monthly_metrics(post_frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    if post_frame.empty:
        return metrics

    prepared = post_frame.copy()
    prepared["single_commenter_thread"] = (
        (prepared["num_comments_observed"] > 0)
        & (prepared["max_depth"] <= 1)
    ).astype(float)
    prepared["multi_actor_thread"] = (prepared["unique_non_op_commenters"] >= 3).astype(float)
    prepared["focused_thread"] = (
        (prepared["non_op_comment_total"] > 0)
        & ((prepared["max_non_op_comment_count"] / prepared["non_op_comment_total"]) > 0.5)
    ).astype(float)
    prepared["distributed_thread"] = (
        (prepared["unique_non_op_commenters"] >= 5)
        & (prepared["non_op_comment_total"] > 0)
        & ((prepared["max_non_op_comment_count"] / prepared["non_op_comment_total"]) <= 0.3)
    ).astype(float)

    grouped = prepared.groupby(["subreddit", "month"], sort=True)
    for (subreddit, month), group in grouped:
        metrics[(str(subreddit), str(month))] = {
            "op_return_rate": float(group["op_returned"].mean()),
            "single_commenter_thread_share": float(group["single_commenter_thread"].mean()),
            "multi_actor_thread_share": float(group["multi_actor_thread"].mean()),
            "focused_thread_share": float(group["focused_thread"].mean()),
            "distributed_thread_share": float(group["distributed_thread"].mean()),
        }
    return metrics


def _dyad_monthly_metrics(reply_edges: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    if reply_edges.empty:
        return metrics

    for (subreddit, month), group in reply_edges.groupby(["subreddit", "month"], sort=True):
        pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
        total_edges = 0
        for row in group.itertuples(index=False):
            source_author = str(row.source_author)
            target_author = str(row.target_author)
            edge_count = int(row.edge_count)
            pair = tuple(sorted((source_author, target_author), key=str.casefold))
            if pair not in pair_stats:
                pair_stats[pair] = {"directions": set(), "edge_count": 0}
            pair_stats[pair]["directions"].add((source_author, target_author))
            pair_stats[pair]["edge_count"] += edge_count
            total_edges += edge_count

        total_pairs = len(pair_stats)
        reciprocal_pairs = sum(1 for stats in pair_stats.values() if len(stats["directions"]) >= 2)
        repeated_pair_edges = sum(
            int(stats["edge_count"])
            for stats in pair_stats.values()
            if int(stats["edge_count"]) >= 2
        )
        metrics[(str(subreddit), str(month))] = {
            "reciprocal_dyad_share": reciprocal_pairs / total_pairs if total_pairs else 0.0,
            "repeat_dyad_share": repeated_pair_edges / total_edges if total_edges else 0.0,
        }

    return metrics


def _zscore(series: pd.Series, active_mask: pd.Series) -> pd.Series:
    result = pd.Series(0.0, index=series.index, dtype=float)
    active_values = series.loc[active_mask].astype(float)
    if active_values.empty:
        return result
    std = float(active_values.std(ddof=0))
    if std == 0.0:
        return result
    mean = float(active_values.mean())
    result.loc[active_mask] = (active_values - mean) / std
    return result


def _plot_outputs(monthly: pd.DataFrame, figure_paths: dict[str, Path]) -> None:
    _plot_type_trend(monthly, "bond_index", "Bond index", figure_paths["bond"])
    _plot_type_trend(monthly, "identity_index", "Identity index", figure_paths["identity"])


def _plot_type_trend(frame: pd.DataFrame, value_col: str, ylabel: str, out_path: Path) -> None:
    plot_frame = (
        frame.loc[frame["community_type"].isin(["health", "general"])]
        .groupby(["month", "community_type"], as_index=False)[value_col]
        .mean()
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
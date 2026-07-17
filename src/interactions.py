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
    months_since,
    stream_zst,
)
from src.thread_prep import ThreadPrepConfig, prepare_thread_partitions

log = logging.getLogger(__name__)

INTERACTIONS_MONTHLY_FILENAME = "interactions-monthly.csv"
INTERACTIONS_METADATA_FILENAME = "interactions-metadata.json"
INTERACTIONS_CACHE_VERSION = 2
GENAI_REFERENCE_MONTH = "2022-11"
AUTHOR_HISTORY_WARMUP_MONTHS = 6
INTERACTIONS_TABLES = frozenset({
    "submissions",
    "post_metrics",
    "helper_counts",
    "author_activity",
    "reply_edges",
})


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
    thread_prep: ThreadPrepConfig | None = None,
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

    if thread_prep is not None and thread_prep.enabled:
        partitioned = prepare_thread_partitions(
            resolved_comment_paths,
            resolved_submission_paths,
            config=thread_prep,
        )
        _run_partitioned_sqlite_pipeline(
            sqlite_path,
            partitioned.submission_partitions,
            partitioned.comment_partitions,
        )
    else:
        submissions = _load_submissions_into_sqlite(sqlite_path, resolved_submission_paths)
        _stream_comments_into_sqlite(sqlite_path, submissions, resolved_comment_paths)
    monthly = _build_monthly_frame(sqlite_path)
    payload = _write_interactions_outputs(
        monthly,
        table_paths=table_paths,
        figure_paths=figure_paths,
        comment_paths=resolved_comment_paths,
        submission_paths=resolved_submission_paths,
        sqlite_path=sqlite_path,
    )

    return InteractionsArtifacts(
        monthly=monthly,
        table_paths=table_paths,
        figure_paths=figure_paths,
        metadata=payload,
    )


def validate_interactions_sqlite(sqlite_path: Path) -> None:
    """Validate the interaction SQLite cache using a read-only connection."""
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Interaction SQLite cache not found: {sqlite_path}")

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            )
        }
        missing_tables = INTERACTIONS_TABLES - table_names
        if missing_tables:
            missing = ", ".join(sorted(missing_tables))
            raise ValueError(f"Interaction SQLite cache is missing tables: {missing}")
        check_result = conn.execute("PRAGMA quick_check(1)").fetchone()
        if check_result != ("ok",):
            raise ValueError(f"Interaction SQLite cache failed quick_check: {check_result}")
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Could not read interaction SQLite cache: {sqlite_path}") from exc
    finally:
        if conn is not None:
            conn.close()


def finalize_interactions_cache(
    *,
    cache_dir: Path | None = None,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> InteractionsArtifacts:
    """Build interaction outputs from an existing SQLite cache without rebuilding it."""
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_cache = cache_dir or (OUTPUT_DIR / "cache")
    sqlite_path = out_cache / "interactions.sqlite"
    validate_interactions_sqlite(sqlite_path)

    resolved_comment_paths = discover_filtered_paths("comments")
    resolved_submission_paths = discover_filtered_paths("submissions")
    table_paths = _table_paths(out_tables)
    figure_paths = _figure_paths(out_figures)
    monthly = _build_monthly_frame(sqlite_path)
    payload = _write_interactions_outputs(
        monthly,
        table_paths=table_paths,
        figure_paths=figure_paths,
        comment_paths=resolved_comment_paths,
        submission_paths=resolved_submission_paths,
        sqlite_path=sqlite_path,
    )
    return InteractionsArtifacts(
        monthly=monthly,
        table_paths=table_paths,
        figure_paths=figure_paths,
        metadata=payload,
    )


def _write_interactions_outputs(
    monthly: pd.DataFrame,
    *,
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
    comment_paths: list[Path],
    submission_paths: list[Path],
    sqlite_path: Path,
) -> dict[str, Any]:
    """Write derived interaction outputs from an assembled monthly frame."""
    table_paths["monthly"].parent.mkdir(parents=True, exist_ok=True)
    figure_paths["bond"].parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(table_paths["monthly"], index=False)
    _plot_outputs(monthly, figure_paths)

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "fingerprint": fingerprint_hash(_source_payload(comment_paths, submission_paths)),
        "sources": _source_payload(comment_paths, submission_paths),
        "sqlite_path": str(sqlite_path),
        "n_rows": int(len(monthly)),
    }
    table_paths["metadata"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


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
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            created_utc INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS post_metrics (
            submission_id TEXT PRIMARY KEY,
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            num_comments_observed INTEGER NOT NULL DEFAULT 0,
            op_returned INTEGER NOT NULL DEFAULT 0,
            deep_reply_count INTEGER NOT NULL DEFAULT 0,
            max_depth INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS helper_counts (
            submission_id TEXT NOT NULL,
            commenter TEXT NOT NULL,
            is_non_op INTEGER NOT NULL,
            comment_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (submission_id, commenter)
        );

        CREATE TABLE IF NOT EXISTS author_activity (
            subreddit TEXT NOT NULL,
            month TEXT NOT NULL,
            author TEXT NOT NULL,
            activity_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (subreddit, month, author)
        );

        CREATE TABLE IF NOT EXISTS reply_edges (
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

    def flush_rows() -> None:
        if not submission_rows:
            return
        with conn:
            conn.executemany(
                "INSERT INTO submissions (submission_id, subreddit, month, author, created_utc) VALUES (?, ?, ?, ?, ?)",
                submission_rows,
            )
            conn.executemany(
                "INSERT INTO post_metrics (submission_id, subreddit, month, author) VALUES (?, ?, ?, ?)",
                post_rows,
            )
        submission_rows.clear()
        post_rows.clear()

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
                if len(submission_rows) >= 100_000:
                    flush_rows()

        flush_rows()

    conn.close()
    return submissions


def _run_partitioned_sqlite_pipeline(
    sqlite_path: Path,
    submission_partitions: list[Path],
    comment_partitions: list[Path],
) -> None:
    if len(submission_partitions) != len(comment_partitions):
        raise ValueError("submission and comment partitions must have the same length")

    for submission_partition, comment_partition in zip(submission_partitions, comment_partitions, strict=True):
        submissions = _load_submissions_into_sqlite(sqlite_path, [submission_partition])
        _stream_comments_into_sqlite(sqlite_path, submissions, [comment_partition])


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
        resolved = stack.pop()
        parent_comment_id = resolved[0]
        parent_subreddit = resolved[2]
        resolved_author = resolved[4]
        resolved_depth = resolved[5]
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
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        months = _sqlite_months(conn)
        subreddits = _sqlite_subreddits(conn)
        if not months or not subreddits:
            return _empty_monthly_frame()

        rows: list[dict[str, Any]] = []
        first_month = months[0]
        for index, subreddit in enumerate(subreddits, start=1):
            log.info("Aggregating interaction metrics for subreddit %d/%d: %s", index, len(subreddits), subreddit)
            author_metrics = _sqlite_author_monthly_metrics(conn, subreddit, months)
            post_metrics = _sqlite_post_monthly_metrics(conn, subreddit)
            dyad_metrics = _sqlite_dyad_monthly_metrics(conn, subreddit)
            for month in months:
                author_row = author_metrics.get((subreddit, month), {})
                post_row = post_metrics.get((subreddit, month), {})
                dyad_row = dyad_metrics.get((subreddit, month), {})
                rows.append({
                    "subreddit": subreddit,
                    "month": month,
                    "community_type": classify_subreddit(subreddit),
                    "unique_authors": int(author_row.get("unique_authors", 0)),
                    "author_history_observed": int(
                        months_since(month, reference=first_month) >= AUTHOR_HISTORY_WARMUP_MONTHS
                    ),
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
    finally:
        conn.close()

    monthly = pd.DataFrame(rows).sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)
    return _add_interaction_indices(monthly)


def _sqlite_months(conn: sqlite3.Connection) -> list[str]:
    """Return the complete month range represented in the SQLite cache."""
    rows = conn.execute(
        """
        SELECT month FROM post_metrics
        UNION
        SELECT month FROM author_activity
        UNION
        SELECT month FROM reply_edges
        ORDER BY month
        """,
    ).fetchall()
    if not rows:
        return []
    return iter_month_range(str(rows[0][0]), str(rows[-1][0]))


def _sqlite_subreddits(conn: sqlite3.Connection) -> list[str]:
    """Return subreddits represented in any interaction SQLite table."""
    rows = conn.execute(
        """
        SELECT subreddit FROM post_metrics
        UNION
        SELECT subreddit FROM author_activity
        UNION
        SELECT subreddit FROM reply_edges
        """,
    ).fetchall()
    return sorted((str(row[0]) for row in rows), key=str.casefold)


def _load_subreddit_post_metrics(conn: sqlite3.Connection, subreddit: str) -> pd.DataFrame:
    """Load one subreddit's post metrics and associated helper summaries."""
    return pd.read_sql_query(
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
                h.submission_id,
                SUM(CASE WHEN h.is_non_op = 1 THEN 1 ELSE 0 END) AS unique_non_op_commenters,
                SUM(CASE WHEN h.is_non_op = 1 THEN h.comment_count ELSE 0 END) AS non_op_comment_total,
                COALESCE(MAX(CASE WHEN h.is_non_op = 1 THEN h.comment_count ELSE 0 END), 0) AS max_non_op_comment_count
            FROM helper_counts h
            JOIN post_metrics related_post ON related_post.submission_id = h.submission_id
            WHERE related_post.subreddit = ?
            GROUP BY h.submission_id
        ) h ON h.submission_id = p.submission_id
        WHERE p.subreddit = ?
        ORDER BY p.month, p.submission_id
        """,
        conn,
        params=(subreddit, subreddit),
    )


def _sqlite_author_monthly_metrics(
    conn: sqlite3.Connection,
    subreddit: str,
    months: list[str],
) -> dict[tuple[str, str], dict[str, float | int]]:
    """Compute author-history metrics in SQLite without materializing authors."""
    rows = conn.execute(
        """
        WITH author_months AS (
            SELECT month, author
            FROM author_activity
            WHERE subreddit = ?
        ), first_months AS (
            SELECT author, MIN(month) AS first_month
            FROM author_months
            GROUP BY author
        ), current_counts AS (
            SELECT month, COUNT(*) AS unique_authors
            FROM author_months
            GROUP BY month
        ), repeat_counts AS (
            SELECT current.month, COUNT(*) AS repeat_authors
            FROM author_months current
            JOIN author_months previous
              ON previous.author = current.author
             AND previous.month = strftime('%Y-%m', date(current.month || '-01', '-1 month'))
            GROUP BY current.month
        ), new_counts AS (
            SELECT month, COUNT(*) AS new_authors
            FROM author_months
            JOIN first_months USING (author)
            WHERE month = first_month
            GROUP BY month
        ), returning_counts AS (
            SELECT current.month, COUNT(*) AS returning_authors
            FROM author_months current
            JOIN first_months first USING (author)
            WHERE current.month > first.first_month
              AND NOT EXISTS (
                  SELECT 1
                  FROM author_months previous
                  WHERE previous.author = current.author
                    AND previous.month = strftime('%Y-%m', date(current.month || '-01', '-1 month'))
              )
            GROUP BY current.month
        )
        SELECT
            months.month,
            COALESCE(current_counts.unique_authors, 0),
            COALESCE(new_counts.new_authors, 0),
            COALESCE(returning_counts.returning_authors, 0),
            COALESCE(repeat_counts.repeat_authors, 0)
        FROM (SELECT ? AS month UNION ALL SELECT month FROM author_months) months
        LEFT JOIN current_counts USING (month)
        LEFT JOIN new_counts USING (month)
        LEFT JOIN returning_counts USING (month)
        LEFT JOIN repeat_counts USING (month)
        GROUP BY months.month
        """,
        (subreddit, months[0]),
    ).fetchall()
    metrics: dict[tuple[str, str], dict[str, float | int]] = {}
    values = {
        str(row[0]): (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
        for row in rows
    }
    for month in months:
        unique_authors, new_authors, returning_authors, repeat_authors = values.get(month, (0, 0, 0, 0))
        denominator = unique_authors or 1
        metrics[(subreddit, month)] = {
            "unique_authors": unique_authors,
            "new_author_share": new_authors / denominator,
            "returning_author_share": returning_authors / denominator,
            "repeat_author_share": repeat_authors / denominator,
        }
    return metrics


def _sqlite_post_monthly_metrics(
    conn: sqlite3.Connection,
    subreddit: str,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute post/thread metrics in SQLite and return only monthly rows."""
    rows = conn.execute(
        """
        WITH helper AS (
            SELECT
                h.submission_id,
                SUM(CASE WHEN h.is_non_op = 1 THEN 1 ELSE 0 END) AS unique_non_op_commenters,
                SUM(CASE WHEN h.is_non_op = 1 THEN h.comment_count ELSE 0 END) AS non_op_comment_total,
                COALESCE(MAX(CASE WHEN h.is_non_op = 1 THEN h.comment_count ELSE 0 END), 0) AS max_non_op_comment_count
            FROM helper_counts h
            JOIN post_metrics p ON p.submission_id = h.submission_id
            WHERE p.subreddit = ?
            GROUP BY h.submission_id
        ), post_values AS (
            SELECT
                p.month,
                p.op_returned,
                p.num_comments_observed,
                p.max_depth,
                COALESCE(h.unique_non_op_commenters, 0) AS unique_non_op_commenters,
                COALESCE(h.non_op_comment_total, 0) AS non_op_comment_total,
                COALESCE(h.max_non_op_comment_count, 0) AS max_non_op_comment_count
            FROM post_metrics p
            LEFT JOIN helper h ON h.submission_id = p.submission_id
            WHERE p.subreddit = ?
        )
        SELECT
            month,
            AVG(op_returned),
            AVG(CASE WHEN num_comments_observed > 0 AND max_depth <= 1 THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN unique_non_op_commenters >= 3 THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN non_op_comment_total > 0
                      AND max_non_op_comment_count * 1.0 / non_op_comment_total > 0.5
                     THEN 1.0 ELSE 0.0 END),
            AVG(CASE WHEN unique_non_op_commenters >= 5
                      AND non_op_comment_total > 0
                      AND max_non_op_comment_count * 1.0 / non_op_comment_total <= 0.3
                     THEN 1.0 ELSE 0.0 END)
        FROM post_values
        GROUP BY month
        """,
        (subreddit, subreddit),
    ).fetchall()
    names = (
        "op_return_rate",
        "single_commenter_thread_share",
        "multi_actor_thread_share",
        "focused_thread_share",
        "distributed_thread_share",
    )
    return {
        (subreddit, str(row[0])): {name: float(value or 0.0) for name, value in zip(names, row[1:], strict=True)}
        for row in rows
    }


def _sqlite_dyad_monthly_metrics(
    conn: sqlite3.Connection,
    subreddit: str,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute reciprocal and repeated dyad shares in SQLite."""
    rows = conn.execute(
        """
        WITH directed AS (
            SELECT month, source_author, target_author, edge_count
            FROM reply_edges
            WHERE subreddit = ?
        ), pairs AS (
            SELECT
                month,
                CASE WHEN source_author < target_author THEN source_author ELSE target_author END AS author_a,
                CASE WHEN source_author < target_author THEN target_author ELSE source_author END AS author_b,
                COUNT(*) AS directions,
                SUM(edge_count) AS pair_edges
            FROM directed
            GROUP BY month, author_a, author_b
        ), totals AS (
            SELECT month, SUM(pair_edges) AS total_edges
            FROM pairs
            GROUP BY month
        )
        SELECT
            pairs.month,
            AVG(CASE WHEN directions >= 2 THEN 1.0 ELSE 0.0 END),
            SUM(CASE WHEN pair_edges >= 2 THEN pair_edges ELSE 0 END) * 1.0 / totals.total_edges
        FROM pairs
        JOIN totals USING (month)
        GROUP BY pairs.month
        """,
        (subreddit,),
    ).fetchall()
    return {
        (subreddit, str(row[0])): {
            "reciprocal_dyad_share": float(row[1] or 0.0),
            "repeat_dyad_share": float(row[2] or 0.0),
        }
        for row in rows
    }


def _empty_monthly_frame() -> pd.DataFrame:
    """Return an empty interaction monthly frame with its public columns."""
    return pd.DataFrame(columns=[
        "subreddit",
        "month",
        "community_type",
        "unique_authors",
        "author_history_observed",
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


def _add_interaction_indices(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add pre-period-standardized bond and identity indices to monthly metrics."""
    active_mask = monthly["unique_authors"] > 0
    reference_mask = (
        active_mask
        & monthly["author_history_observed"].astype(bool)
        & (monthly["month"] < GENAI_REFERENCE_MONTH)
    )
    if not reference_mask.any():
        reference_mask = active_mask & (monthly["month"] < GENAI_REFERENCE_MONTH)
    monthly["bond_index"] = (
        _zscore(monthly["op_return_rate"], active_mask, reference_mask)
        + _zscore(monthly["reciprocal_dyad_share"], active_mask, reference_mask)
        + _zscore(monthly["repeat_dyad_share"], active_mask, reference_mask)
        + _zscore(monthly["repeat_author_share"], active_mask, reference_mask)
    )
    monthly["identity_index"] = (
        _zscore(monthly["unique_authors"].astype(float), active_mask, reference_mask)
        + _zscore(monthly["new_author_share"], active_mask, reference_mask)
        + _zscore(monthly["distributed_thread_share"], active_mask, reference_mask)
        - _zscore(monthly["focused_thread_share"], active_mask, reference_mask)
    )
    monthly.loc[~active_mask, ["bond_index", "identity_index"]] = 0.0
    return monthly


def _assemble_monthly_frame(
    post_frame: pd.DataFrame,
    author_activity: pd.DataFrame,
    reply_edges: pd.DataFrame,
) -> pd.DataFrame:
    months = _all_months(post_frame, author_activity, reply_edges)
    subreddits = _all_subreddits(post_frame, author_activity, reply_edges)
    if not months or not subreddits:
        return _empty_monthly_frame()

    author_metrics = _author_history_metrics(author_activity, subreddits, months)
    post_metrics = _post_monthly_metrics(post_frame)
    dyad_metrics = _dyad_monthly_metrics(reply_edges)

    rows: list[dict[str, Any]] = []
    first_month = months[0]
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
                "author_history_observed": int(
                    months_since(month, reference=first_month) >= AUTHOR_HISTORY_WARMUP_MONTHS
                ),
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
    return _add_interaction_indices(monthly)


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


def _zscore(
    series: pd.Series,
    active_mask: pd.Series,
    reference_mask: pd.Series,
) -> pd.Series:
    result = pd.Series(0.0, index=series.index, dtype=float)
    reference_values = series.loc[reference_mask].astype(float)
    if reference_values.empty:
        return result
    std = float(reference_values.std(ddof=0))
    if std == 0.0:
        return result
    mean = float(reference_values.mean())
    result.loc[active_mask] = (series.loc[active_mask].astype(float) - mean) / std
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
    plot_frame["month_dt"] = pd.to_datetime(plot_frame["month"], format="%Y-%m")
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
            subset["month_dt"],
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
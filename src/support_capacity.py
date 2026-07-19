"""Measure rolling helper capacity, newcomer reception, and contributor retention."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import OUTPUT_DIR, TABLES_DIR
from src.did import BASELINE_CUTOFF, TRANSITION_MONTHS
from src.helpers import classify_subreddit
from src.io_utils import iter_month_range

log = logging.getLogger(__name__)

DEFAULT_WINDOW_MONTHS = 12
RETENTION_HORIZONS = (1, 3, 6)


@dataclass
class SupportCapacityArtifacts:
    """Tables written by the support-capacity analysis."""

    monthly: pd.DataFrame
    community_summary: pd.DataFrame
    table_paths: dict[str, Path]


def build_support_capacity_monthly(
    author_activity: pd.DataFrame,
    submissions: pd.DataFrame,
    post_metrics: pd.DataFrame,
    helper_counts: pd.DataFrame,
    *,
    window_months: int = DEFAULT_WINDOW_MONTHS,
) -> pd.DataFrame:
    """Build a complete subreddit-month support-capacity panel from cache tables."""
    if window_months <= 0:
        raise ValueError("window_months must be positive")
    required = {
        "author_activity": {"subreddit", "month", "author"},
        "submissions": {"submission_id", "subreddit", "month", "author"},
        "post_metrics": {"submission_id", "num_comments_observed"},
        "helper_counts": {"submission_id", "commenter", "is_non_op", "comment_count"},
    }
    frames = {
        "author_activity": author_activity,
        "submissions": submissions,
        "post_metrics": post_metrics,
        "helper_counts": helper_counts,
    }
    for name, columns in required.items():
        missing = columns - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}")

    activities = author_activity.copy()
    activities = activities.loc[activities["author"].notna()].copy()
    activities["author"] = activities["author"].astype(str)
    activities = activities.loc[activities["author"].ne("")].copy()
    first_month = activities.groupby(["subreddit", "author"])["month"].min().to_dict()
    author_sets = {
        key: set(group["author"])
        for key, group in activities.groupby(["subreddit", "month"], sort=True)
    }

    helpers = helper_counts.loc[helper_counts["is_non_op"] == 1].merge(
        submissions[["submission_id", "subreddit", "month"]],
        on="submission_id",
        how="inner",
    )
    helper_sets = {
        key: set(group["commenter"].astype(str))
        for key, group in helpers.groupby(["subreddit", "month"], sort=True)
    }
    helper_weights = {
        key: group.groupby("commenter")["comment_count"].sum().astype(float).to_dict()
        for key, group in helpers.groupby(["subreddit", "month"], sort=True)
    }

    posts = submissions.merge(
        post_metrics[["submission_id", "num_comments_observed"]],
        on="submission_id",
        how="left",
    )
    posts["num_comments_observed"] = posts["num_comments_observed"].fillna(0)
    rows: list[dict[str, Any]] = []
    for subreddit, group in activities.groupby("subreddit", sort=True):
        months = iter_month_range(str(group["month"].min()), str(group["month"].max()))
        for index, month in enumerate(months):
            current_authors = author_sets.get((subreddit, month), set())
            new_authors = {
                author for author in current_authors if first_month[(subreddit, author)] == month
            }
            current_helpers = helper_sets.get((subreddit, month), set())
            window_start = max(0, index - window_months + 1)
            window = months[window_start:index + 1]
            rolling_helpers = set().union(*(helper_sets.get((subreddit, item), set()) for item in window))
            rolling_weights: dict[str, float] = {}
            for item in window:
                for author, count in helper_weights.get((subreddit, item), {}).items():
                    rolling_weights[author] = rolling_weights.get(author, 0.0) + count
            previous_window = months[max(0, index - window_months):index]
            previous_helpers = set().union(
                *(helper_sets.get((subreddit, item), set()) for item in previous_window)
            ) if previous_window else set()

            row: dict[str, Any] = {
                "subreddit": subreddit,
                "month": month,
                "community_type": classify_subreddit(str(subreddit)),
                "active_contributors": len(current_authors),
                "new_contributors": len(new_authors),
                "newcomer_share": _share(len(new_authors), len(current_authors)),
                "active_helpers": len(current_helpers),
                "rolling_window_months": window_months,
                "rolling_active_helpers": len(rolling_helpers),
                "rolling_effective_helpers": _effective_count(rolling_weights),
                "rolling_top5_helper_share": _top_k_share(rolling_weights, 5),
                "rolling_helper_jaccard": _jaccard(rolling_helpers, previous_helpers),
            }
            newcomer_posts = posts.loc[
                (posts["subreddit"] == subreddit)
                & (posts["month"] == month)
                & (posts["author"].isin(new_authors)),
            ]
            row["newcomer_submissions"] = int(len(newcomer_posts))
            row["newcomer_reply_rate"] = _share(
                int((newcomer_posts["num_comments_observed"] > 0).sum()),
                len(newcomer_posts),
            )
            for horizon in RETENTION_HORIZONS:
                future_month = _month_offset(month, horizon)
                future_authors = author_sets.get((subreddit, future_month), set())
                future_helpers = helper_sets.get((subreddit, future_month), set())
                row[f"newcomer_return_{horizon}m"] = _share(len(new_authors & future_authors), len(new_authors))
                row[f"contributor_return_{horizon}m"] = _share(
                    len(current_authors & future_authors), len(current_authors),
                )
                row[f"helper_return_{horizon}m"] = _share(
                    len(current_helpers & future_helpers), len(current_helpers),
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)


def summarize_support_capacity(monthly: pd.DataFrame) -> pd.DataFrame:
    """Summarize analysis measures by community type and pre/post period."""
    frame = monthly.loc[~monthly["month"].isin(TRANSITION_MONTHS)].copy()
    frame = frame.loc[frame["community_type"].isin(["health", "general"])]
    frame["period"] = frame["month"].map(
        lambda month: "post" if month >= BASELINE_CUTOFF else "pre",
    )
    value_columns = [
        "active_contributors",
        "newcomer_share",
        "rolling_active_helpers",
        "rolling_effective_helpers",
        "rolling_top5_helper_share",
        "rolling_helper_jaccard",
        "newcomer_reply_rate",
        *[f"newcomer_return_{horizon}m" for horizon in RETENTION_HORIZONS],
        *[f"contributor_return_{horizon}m" for horizon in RETENTION_HORIZONS],
        *[f"helper_return_{horizon}m" for horizon in RETENTION_HORIZONS],
    ]
    return frame.groupby(["community_type", "period"], as_index=False)[value_columns].mean()


def run_support_capacity_analysis(
    sqlite_path: Path | None = None,
    *,
    tables_dir: Path | None = None,
    window_months: int = DEFAULT_WINDOW_MONTHS,
) -> SupportCapacityArtifacts:
    """Stream SQL aggregates from the interaction cache into analysis tables."""
    resolved_sqlite = sqlite_path or (OUTPUT_DIR / "cache" / "interactions.sqlite")
    if window_months <= 0:
        raise ValueError("window_months must be positive")
    _validate_support_capacity_cache(resolved_sqlite)
    monthly = _build_support_capacity_from_sqlite(resolved_sqlite, window_months=window_months)
    summary = summarize_support_capacity(monthly)
    out_tables = tables_dir or TABLES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    suffix = "" if window_months == DEFAULT_WINDOW_MONTHS else f"-{window_months}m"
    paths = {
        "monthly": out_tables / f"support-capacity{suffix}-monthly.csv",
        "community_summary": out_tables / f"support-capacity{suffix}-community-summary.csv",
        "metadata": out_tables / f"support-capacity{suffix}-metadata.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    summary.to_csv(paths["community_summary"], index=False)
    paths["metadata"].write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sqlite_path": str(resolved_sqlite),
        "window_months": window_months,
        "retention_horizons": list(RETENTION_HORIZONS),
        "cutoff_month": BASELINE_CUTOFF,
        "excluded_transition_months": sorted(TRANSITION_MONTHS),
    }, indent=2), encoding="utf-8")
    log.info("Wrote %s", paths["monthly"])
    return SupportCapacityArtifacts(monthly=monthly, community_summary=summary, table_paths=paths)


def _build_support_capacity_from_sqlite(sqlite_path: Path, *, window_months: int) -> pd.DataFrame:
    """Build monthly measures with bounded memory from SQLite aggregates."""
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        subreddits = [str(row[0]) for row in conn.execute(
            "SELECT DISTINCT subreddit FROM author_activity ORDER BY subreddit",
        )]
        rows: list[dict[str, Any]] = []
        for index, subreddit in enumerate(subreddits, start=1):
            log.info("Aggregating support capacity for subreddit %d/%d: %s", index, len(subreddits), subreddit)
            rows.extend(_sqlite_subreddit_rows(conn, subreddit, window_months=window_months))
    finally:
        conn.close()
    return pd.DataFrame(rows).sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)


def _validate_support_capacity_cache(sqlite_path: Path) -> None:
    """Confirm that a readable cache exposes the tables used by this analysis."""
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"Interaction SQLite cache not found: {sqlite_path}")
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        required = {"author_activity", "submissions", "post_metrics", "helper_counts"}
        missing = required - tables
        if missing:
            raise ValueError(f"Interaction SQLite cache is missing tables: {', '.join(sorted(missing))}")
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Could not read interaction SQLite cache: {sqlite_path}") from exc
    finally:
        conn.close()


def _sqlite_subreddit_rows(
    conn: sqlite3.Connection,
    subreddit: str,
    *,
    window_months: int,
) -> list[dict[str, Any]]:
    """Return one subreddit's monthly rows from compact SQL aggregates."""
    months = _sqlite_subreddit_months(conn, subreddit)
    if not months:
        return []
    author_metrics = _sqlite_author_metrics(conn, subreddit)
    newcomer_posts = _sqlite_newcomer_posts(conn, subreddit)
    helper_returns = _sqlite_helper_returns(conn, subreddit)
    helper_months = iter(_iter_helper_months(conn, subreddit))
    next_helper_month = next(helper_months, None)
    helper_windows: deque[dict[str, float]] = deque()
    rolling_weights: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for index, month in enumerate(months):
        current_helpers: dict[str, float] = {}
        if next_helper_month is not None and next_helper_month[0] == month:
            current_helpers = next_helper_month[1]
            next_helper_month = next(helper_months, None)
        helper_windows.append(current_helpers)
        rolling_weights.update(current_helpers)
        expired_helpers: dict[str, float] = {}
        if len(helper_windows) > window_months:
            expired_helpers = helper_windows.popleft()
            _subtract_weights(rolling_weights, expired_helpers)
        previous_helper_weights = Counter(rolling_weights)
        _subtract_weights(previous_helper_weights, current_helpers)
        if index >= window_months:
            previous_helper_weights.update(expired_helpers)
        author_row = author_metrics.get(month, {})
        post_row = newcomer_posts.get(month, {})
        row: dict[str, Any] = {
            "subreddit": subreddit,
            "month": month,
            "community_type": classify_subreddit(subreddit),
            "active_contributors": int(author_row.get("active_contributors", 0)),
            "new_contributors": int(author_row.get("new_contributors", 0)),
            "newcomer_share": float(author_row.get("newcomer_share", 0.0)),
            "active_helpers": len(current_helpers),
            "rolling_window_months": window_months,
            "rolling_active_helpers": len(rolling_weights),
            "rolling_effective_helpers": _effective_count(dict(rolling_weights)),
            "rolling_top5_helper_share": _top_k_share(dict(rolling_weights), 5),
            "rolling_helper_jaccard": _jaccard(set(rolling_weights), set(previous_helper_weights)),
            "newcomer_submissions": int(post_row.get("newcomer_submissions", 0)),
            "newcomer_reply_rate": float(post_row.get("newcomer_reply_rate", 0.0)),
        }
        for horizon in RETENTION_HORIZONS:
            row[f"newcomer_return_{horizon}m"] = float(
                author_row.get(f"newcomer_return_{horizon}m", 0.0),
            )
            row[f"contributor_return_{horizon}m"] = float(
                author_row.get(f"contributor_return_{horizon}m", 0.0),
            )
            row[f"helper_return_{horizon}m"] = float(
                helper_returns.get(month, {}).get(f"helper_return_{horizon}m", 0.0),
            )
        rows.append(row)
    return rows


def _sqlite_subreddit_months(conn: sqlite3.Connection, subreddit: str) -> list[str]:
    """Return a continuous month range for one subreddit."""
    row = conn.execute(
        "SELECT MIN(month), MAX(month) FROM author_activity WHERE subreddit = ?",
        (subreddit,),
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return []
    return iter_month_range(str(row[0]), str(row[1]))


def _sqlite_author_metrics(conn: sqlite3.Connection, subreddit: str) -> dict[str, dict[str, float | int]]:
    """Compute contributor and newcomer retention measures entirely in SQLite."""
    horizon_columns = ",\n            ".join(
        f"""CAST(SUM(CASE WHEN EXISTS (
                SELECT 1 FROM author_months future
                WHERE future.author = current.author
                  AND future.month = strftime('%Y-%m', date(current.month || '-01', '+{horizon} months'))
            ) THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS contributor_return_{horizon}m,
            CAST(SUM(CASE WHEN current.month = first_month AND EXISTS (
                SELECT 1 FROM author_months future
                WHERE future.author = current.author
                  AND future.month = strftime('%Y-%m', date(current.month || '-01', '+{horizon} months'))
            ) THEN 1 ELSE 0 END) AS REAL) /
                NULLIF(SUM(CASE WHEN current.month = first_month THEN 1 ELSE 0 END), 0) AS newcomer_return_{horizon}m"""
        for horizon in RETENTION_HORIZONS
    )
    rows = conn.execute(
        f"""
        WITH author_months AS (
            SELECT month, author FROM author_activity WHERE subreddit = ?
        ), first_months AS (
            SELECT author, MIN(month) AS first_month FROM author_months GROUP BY author
        )
        SELECT
            current.month,
            COUNT(*) AS active_contributors,
            SUM(CASE WHEN current.month = first_month THEN 1 ELSE 0 END) AS new_contributors,
            CAST(SUM(CASE WHEN current.month = first_month THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS newcomer_share,
            {horizon_columns}
        FROM author_months current
        JOIN first_months USING (author)
        GROUP BY current.month
        ORDER BY current.month
        """,
        (subreddit,),
    ).fetchall()
    columns = ["month", "active_contributors", "new_contributors", "newcomer_share"]
    for horizon in RETENTION_HORIZONS:
        columns.extend([f"contributor_return_{horizon}m", f"newcomer_return_{horizon}m"])
    return {
        str(row[0]): {
            column: 0.0 if value is None else value
            for column, value in zip(columns[1:], row[1:], strict=True)
        }
        for row in rows
    }


def _sqlite_newcomer_posts(conn: sqlite3.Connection, subreddit: str) -> dict[str, dict[str, float | int]]:
    """Aggregate reply receipt for submissions made in a user's first month."""
    rows = conn.execute(
        """
        WITH first_months AS (
            SELECT author, MIN(month) AS first_month
            FROM author_activity
            WHERE subreddit = ?
            GROUP BY author
        )
        SELECT
            s.month,
            COUNT(*) AS newcomer_submissions,
            CAST(SUM(CASE WHEN p.num_comments_observed > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS newcomer_reply_rate
        FROM submissions s
        JOIN first_months f ON f.author = s.author AND f.first_month = s.month
        LEFT JOIN post_metrics p ON p.submission_id = s.submission_id
        WHERE s.subreddit = ?
        GROUP BY s.month
        ORDER BY s.month
        """,
        (subreddit, subreddit),
    ).fetchall()
    return {
        str(month): {
            "newcomer_submissions": int(count),
            "newcomer_reply_rate": float(rate),
        }
        for month, count, rate in rows
    }


def _sqlite_helper_returns(conn: sqlite3.Connection, subreddit: str) -> dict[str, dict[str, float]]:
    """Compute helper retention at each horizon from distinct helper-month rows."""
    horizons = ",\n            ".join(
        f"""CAST(SUM(CASE WHEN EXISTS (
                SELECT 1 FROM helper_months future
                WHERE future.commenter = current.commenter
                  AND future.month = strftime('%Y-%m', date(current.month || '-01', '+{horizon} months'))
            ) THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS helper_return_{horizon}m"""
        for horizon in RETENTION_HORIZONS
    )
    rows = conn.execute(
        f"""
        WITH helper_months AS (
            SELECT s.month, h.commenter
            FROM helper_counts h
            JOIN submissions s ON s.submission_id = h.submission_id
            WHERE s.subreddit = ? AND h.is_non_op = 1
            GROUP BY s.month, h.commenter
        )
        SELECT current.month, {horizons}
        FROM helper_months current
        GROUP BY current.month
        ORDER BY current.month
        """,
        (subreddit,),
    ).fetchall()
    columns = [f"helper_return_{horizon}m" for horizon in RETENTION_HORIZONS]
    return {
        str(row[0]): {
            column: 0.0 if value is None else float(value)
            for column, value in zip(columns, row[1:], strict=True)
        }
        for row in rows
    }


def _iter_helper_months(
    conn: sqlite3.Connection,
    subreddit: str,
) -> Iterator[tuple[str, dict[str, float]]]:
    """Read helper activity one grouped month at a time for bounded rolling state."""
    rows = conn.execute(
        """
        SELECT s.month, h.commenter, SUM(h.comment_count)
        FROM helper_counts h
        JOIN submissions s ON s.submission_id = h.submission_id
        WHERE s.subreddit = ? AND h.is_non_op = 1
        GROUP BY s.month, h.commenter
        ORDER BY s.month, h.commenter
        """,
        (subreddit,),
    )
    current_month: str | None = None
    weights: dict[str, float] = {}
    for month, commenter, count in rows:
        normalized_month = str(month)
        if current_month is not None and normalized_month != current_month:
            yield current_month, weights
            weights = {}
        current_month = normalized_month
        weights[str(commenter)] = float(count)
    if current_month is not None:
        yield current_month, weights


def _subtract_weights(target: Counter[str], values: dict[str, float]) -> None:
    """Remove one month's helper weights and discard zero-count helpers."""
    for author, count in values.items():
        target[author] -= count
        if target[author] <= 0:
            del target[author]


def _month_offset(month: str, offset: int) -> str:
    """Return the canonical month string offset from ``month``."""
    year, number = (int(value) for value in month.split("-"))
    total = year * 12 + number - 1 + offset
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _share(numerator: int, denominator: int) -> float:
    """Return a zero-safe proportion."""
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set[str], right: set[str]) -> float:
    """Return Jaccard overlap, using zero for two empty sets."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _top_k_share(weights: dict[str, float], top_k: int) -> float:
    """Return the activity share of the most active ``top_k`` helpers."""
    total = sum(weights.values())
    return sum(sorted(weights.values(), reverse=True)[:top_k]) / total if total else 0.0


def _effective_count(weights: dict[str, float]) -> float:
    """Return the inverse-HHI effective number of rolling helpers."""
    total = sum(weights.values())
    if not total:
        return 0.0
    hhi = sum((count / total) ** 2 for count in weights.values())
    return 1.0 / hhi if hhi else 0.0
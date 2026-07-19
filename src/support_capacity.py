"""Measure rolling helper capacity, newcomer reception, and contributor retention."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUT_DIR, TABLES_DIR
from src.did import BASELINE_CUTOFF, TRANSITION_MONTHS
from src.helpers import classify_subreddit
from src.interactions import validate_interactions_sqlite
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
    """Read the interaction cache and write support-capacity analysis tables."""
    resolved_sqlite = sqlite_path or (OUTPUT_DIR / "cache" / "interactions.sqlite")
    validate_interactions_sqlite(resolved_sqlite)
    tables = _load_cache_tables(resolved_sqlite)
    monthly = build_support_capacity_monthly(**tables, window_months=window_months)
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


def _load_cache_tables(sqlite_path: Path) -> dict[str, pd.DataFrame]:
    """Load only the interaction-cache columns needed for support capacity."""
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        return {
            "author_activity": pd.read_sql_query(
                "SELECT subreddit, month, author FROM author_activity", conn,
            ),
            "submissions": pd.read_sql_query(
                "SELECT submission_id, subreddit, month, author FROM submissions", conn,
            ),
            "post_metrics": pd.read_sql_query(
                "SELECT submission_id, num_comments_observed FROM post_metrics", conn,
            ),
            "helper_counts": pd.read_sql_query(
                "SELECT submission_id, commenter, is_non_op, comment_count FROM helper_counts", conn,
            ),
        }
    finally:
        conn.close()


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
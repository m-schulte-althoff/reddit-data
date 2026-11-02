"""Tests for rolling helper-capacity and retention metrics."""

from __future__ import annotations

import sqlite3
import io
import pandas as pd
import pytest
from pathlib import Path

from src.support_capacity import (
    _build_support_capacity_from_sqlite,
    build_support_capacity_monthly,
    run_support_capacity_analysis,
    summarize_support_capacity,
)


def test_support_capacity_tracks_helpers_newcomers_and_returns() -> None:
    author_activity = pd.DataFrame([
        {"subreddit": "Anxiety", "month": "2022-09", "author": "alex"},
        {"subreddit": "Anxiety", "month": "2022-10", "author": "alex"},
        {"subreddit": "Anxiety", "month": "2022-10", "author": "bea"},
        {"subreddit": "Anxiety", "month": "2022-12", "author": "bea"},
        {"subreddit": "Anxiety", "month": "2022-12", "author": "casey"},
        {"subreddit": "Anxiety", "month": "2023-01", "author": "casey"},
    ])
    submissions = pd.DataFrame([
        {"submission_id": "p1", "subreddit": "Anxiety", "month": "2022-10", "author": "bea"},
        {"submission_id": "p2", "subreddit": "Anxiety", "month": "2022-12", "author": "casey"},
    ])
    post_metrics = pd.DataFrame([
        {"submission_id": "p1", "num_comments_observed": 2},
        {"submission_id": "p2", "num_comments_observed": 0},
    ])
    helper_counts = pd.DataFrame([
        {"submission_id": "p1", "commenter": "alex", "is_non_op": 1, "comment_count": 2},
        {"submission_id": "p2", "commenter": "bea", "is_non_op": 1, "comment_count": 1},
    ])

    monthly = build_support_capacity_monthly(
        author_activity,
        submissions,
        post_metrics,
        helper_counts,
        window_months=2,
    )

    october = monthly.loc[monthly["month"] == "2022-10"].iloc[0]
    december = monthly.loc[monthly["month"] == "2022-12"].iloc[0]
    assert october["new_contributors"] == 1
    assert october["newcomer_reply_rate"] == pytest.approx(1.0)
    assert october["newcomer_return_3m"] == pytest.approx(0.0)
    assert december["newcomer_reply_rate"] == pytest.approx(0.0)
    assert december["newcomer_return_1m"] == pytest.approx(1.0)
    assert december["rolling_active_helpers"] == 1
    assert december["rolling_effective_helpers"] == pytest.approx(1.0)


@pytest.mark.parametrize("backend", ["dataframe", "sqlite"])
def test_retention_masks_followup_beyond_global_coverage(tmp_path: Path, backend: str) -> None:
    """Unavailable follow-up is missing; observed non-return and boundary months remain valid."""
    activity = pd.DataFrame([
        {"subreddit": "Anxiety", "month": month, "author": "alice"}
        for month in ("2025-06", "2025-09", "2025-11", "2025-12")
    ] + [{"subreddit": "Asthma", "month": "2025-06", "author": "bob"}])
    submissions = pd.DataFrame([
        {"submission_id": str(index), "subreddit": row.subreddit, "month": row.month, "author": "op"}
        for index, row in enumerate(activity.itertuples())
    ])
    posts = pd.DataFrame([
        {"submission_id": row.submission_id, "num_comments_observed": 1}
        for row in submissions.itertuples()
    ])
    helpers = pd.DataFrame([
        {"submission_id": str(index), "commenter": row.author, "is_non_op": 1, "comment_count": 1}
        for index, row in enumerate(activity.itertuples())
    ])
    if backend == "dataframe":
        monthly = build_support_capacity_monthly(activity, submissions, posts, helpers)
    else:
        path = tmp_path / "cache.sqlite"
        with sqlite3.connect(path) as conn:
            for name, frame in [("author_activity", activity), ("submissions", submissions),
                                ("post_metrics", posts), ("helper_counts", helpers)]:
                frame.to_sql(name, conn, index=False)
        monthly = _build_support_capacity_from_sqlite(path, window_months=12)
    anxiety = monthly.loc[monthly["subreddit"] == "Anxiety"].set_index("month")
    for horizon, boundary in [(1, "2025-11"), (3, "2025-09"), (6, "2025-06")]:
        for group in ("newcomer", "contributor", "helper"):
            column = f"{group}_return_{horizon}m"
            assert pd.notna(anxiety.loc[boundary, column])
            assert anxiety.loc[anxiety.index > boundary, column].isna().all()
        assert anxiety.loc[boundary, f"contributor_return_{horizon}m"] == pytest.approx(1.0)
        assert anxiety.loc[boundary, f"helper_return_{horizon}m"] == pytest.approx(1.0)
    asthma = monthly.loc[monthly["subreddit"] == "Asthma"].iloc[0]
    assert asthma["contributor_return_6m"] == 0.0
    assert asthma["helper_return_6m"] == 0.0
    summary = summarize_support_capacity(monthly)
    assert summary.loc[summary["period"] == "post", "contributor_return_6m"].iloc[0] == pytest.approx(0.5)
    exported = pd.read_csv(io.StringIO(monthly.to_csv(index=False)))
    assert exported.loc[exported["month"] == "2025-12", "contributor_return_1m"].isna().all()


def test_support_capacity_summary_omits_transition_month() -> None:
    monthly = pd.DataFrame([
        {
            "subreddit": "Anxiety", "month": "2022-10", "community_type": "health",
            "active_contributors": 2, "newcomer_share": 0.5, "rolling_active_helpers": 1,
            "rolling_effective_helpers": 1.0, "rolling_top5_helper_share": 1.0,
            "rolling_helper_jaccard": 0.0, "newcomer_reply_rate": 1.0,
            **{f"{prefix}_{horizon}m": 0.5 for prefix in ("newcomer_return", "contributor_return", "helper_return") for horizon in (1, 3, 6)},
        },
        {
            "subreddit": "Anxiety", "month": "2022-11", "community_type": "health",
            "active_contributors": 999, "newcomer_share": 1.0, "rolling_active_helpers": 999,
            "rolling_effective_helpers": 999.0, "rolling_top5_helper_share": 1.0,
            "rolling_helper_jaccard": 1.0, "newcomer_reply_rate": 1.0,
            **{f"{prefix}_{horizon}m": 1.0 for prefix in ("newcomer_return", "contributor_return", "helper_return") for horizon in (1, 3, 6)},
        },
    ])

    summary = summarize_support_capacity(monthly)

    assert summary.loc[summary["period"] == "pre", "active_contributors"].iloc[0] == 2


def test_nondefault_window_uses_distinct_output_filenames(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "interactions.sqlite"
    conn = sqlite3.connect(sqlite_path)
    conn.executescript("""
        CREATE TABLE author_activity (subreddit TEXT, month TEXT, author TEXT);
        CREATE TABLE submissions (submission_id TEXT, subreddit TEXT, month TEXT, author TEXT);
        CREATE TABLE post_metrics (submission_id TEXT, num_comments_observed INTEGER);
        CREATE TABLE helper_counts (submission_id TEXT, commenter TEXT, is_non_op INTEGER, comment_count INTEGER);
    """)
    conn.execute("INSERT INTO author_activity VALUES ('Anxiety', '2022-10', 'alex')")
    conn.commit()
    conn.close()

    result = run_support_capacity_analysis(
        sqlite_path=sqlite_path,
        tables_dir=tmp_path / "tables",
        window_months=24,
    )

    assert result.table_paths["monthly"].name == "support-capacity-24m-monthly.csv"
    conn = sqlite3.connect(sqlite_path)
    index_names = {row[1] for row in conn.execute("PRAGMA index_list('submissions')")}
    conn.close()
    assert "idx_submissions_support_capacity" in index_names


def test_support_capacity_streams_sqlite_cache_metrics(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "interactions.sqlite"
    conn = sqlite3.connect(sqlite_path)
    conn.executescript("""
        CREATE TABLE author_activity (subreddit TEXT, month TEXT, author TEXT);
        CREATE TABLE submissions (submission_id TEXT, subreddit TEXT, month TEXT, author TEXT);
        CREATE TABLE post_metrics (submission_id TEXT, num_comments_observed INTEGER);
        CREATE TABLE helper_counts (submission_id TEXT, commenter TEXT, is_non_op INTEGER, comment_count INTEGER);
    """)
    conn.executemany("INSERT INTO author_activity VALUES (?, ?, ?)", [
        ("Anxiety", "2022-09", "alex"),
        ("Anxiety", "2022-10", "alex"),
        ("Anxiety", "2022-10", "bea"),
        ("Anxiety", "2022-12", "bea"),
        ("Anxiety", "2022-12", "casey"),
        ("Anxiety", "2023-01", "casey"),
    ])
    conn.executemany("INSERT INTO submissions VALUES (?, ?, ?, ?)", [
        ("p1", "Anxiety", "2022-10", "bea"),
        ("p2", "Anxiety", "2022-12", "casey"),
    ])
    conn.executemany("INSERT INTO post_metrics VALUES (?, ?)", [("p1", 2), ("p2", 0)])
    conn.executemany("INSERT INTO helper_counts VALUES (?, ?, ?, ?)", [
        ("p1", "alex", 1, 2),
        ("p2", "bea", 1, 1),
    ])
    conn.commit()
    conn.close()

    result = run_support_capacity_analysis(
        sqlite_path=sqlite_path,
        tables_dir=tmp_path / "tables",
        window_months=2,
    )

    october = result.monthly.loc[result.monthly["month"] == "2022-10"].iloc[0]
    december = result.monthly.loc[result.monthly["month"] == "2022-12"].iloc[0]
    assert october["newcomer_reply_rate"] == pytest.approx(1.0)
    assert december["newcomer_reply_rate"] == pytest.approx(0.0)
    assert december["newcomer_return_1m"] == pytest.approx(1.0)
    assert december["rolling_active_helpers"] == 1

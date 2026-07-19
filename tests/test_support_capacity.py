"""Tests for rolling helper-capacity and retention metrics."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.support_capacity import (
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


def test_nondefault_window_uses_distinct_output_filenames(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.support_capacity.validate_interactions_sqlite", lambda _: None)
    monkeypatch.setattr("src.support_capacity._load_cache_tables", lambda _: {
        "author_activity": pd.DataFrame([
            {"subreddit": "Anxiety", "month": "2022-10", "author": "alex"},
        ]),
        "submissions": pd.DataFrame(columns=["submission_id", "subreddit", "month", "author"]),
        "post_metrics": pd.DataFrame(columns=["submission_id", "num_comments_observed"]),
        "helper_counts": pd.DataFrame(columns=["submission_id", "commenter", "is_non_op", "comment_count"]),
    })

    result = run_support_capacity_analysis(
        sqlite_path=tmp_path / "interactions.sqlite",
        tables_dir=tmp_path / "tables",
        window_months=24,
    )

    assert result.table_paths["monthly"].name == "support-capacity-24m-monthly.csv"
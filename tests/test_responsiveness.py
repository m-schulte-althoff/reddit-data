"""Tests for src.responsiveness — post-level and monthly responsiveness metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.responsiveness import load_responsiveness_cache, run_responsiveness_analysis
from src.thread_prep import normalize_thread_prep_config
from tests.conftest import write_zst_jsonl


SUBMISSIONS = [
    {
        "id": "s1",
        "subreddit": "askreddit",
        "author": "op1",
        "title": "General help",
        "created_utc": 1664600000,
    },
    {
        "id": "s2",
        "subreddit": "depression",
        "author": "op2",
        "title": "Need support",
        "created_utc": 1664600000,
    },
    {
        "id": "s3",
        "subreddit": "depression",
        "author": "op3",
        "title": "Another post",
        "created_utc": 1664600600,
    },
]

COMMENTS = [
    {
        "id": "c1",
        "subreddit": "askreddit",
        "author": "helper1",
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
        "created_utc": 1664600600,
    },
    {
        "id": "c2",
        "subreddit": "askreddit",
        "author": "op1",
        "parent_id": "t1_c1",
        "link_id": "t3_s1",
        "created_utc": 1664601200,
    },
    {
        "id": "c3",
        "subreddit": "askreddit",
        "author": "helper1",
        "parent_id": "t1_c2",
        "link_id": "t3_s1",
        "created_utc": 1664601500,
    },
    {
        "id": "c4",
        "subreddit": "depression",
        "author": "helper2",
        "parent_id": "t3_s2",
        "link_id": "t3_s2",
        "created_utc": 1664600300,
    },
]


def _make_processed(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    cache_dir = tmp_path / "cache"
    comments_path = processed_dir / "filter-comments-test.jsonl.zst"
    submissions_path = processed_dir / "filter-submissions-test.jsonl.zst"
    write_zst_jsonl(comments_path, COMMENTS)
    write_zst_jsonl(submissions_path, SUBMISSIONS)
    return comments_path, submissions_path, tables_dir, figures_dir, cache_dir


def test_run_responsiveness_analysis_computes_post_and_monthly_metrics(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir, cache_dir = _make_processed(tmp_path)

    result = run_responsiveness_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
    )

    posts = result.posts.set_index("submission_id")
    s1 = posts.loc["s1"]
    assert s1["num_comments_observed"] == 3
    assert s1["has_reply"] == 1
    assert s1["first_reply_latency_minutes"] == 10.0
    assert s1["unique_commenters"] == 2
    assert s1["unique_non_op_commenters"] == 1
    assert s1["op_followup_comments"] == 1
    assert round(float(s1["op_followup_share"]), 4) == round(1 / 3, 4)
    assert s1["direct_reply_count"] == 1
    assert s1["deep_reply_count"] == 2
    assert s1["max_depth"] == 3
    assert round(float(s1["threading_ratio"]), 4) == round(2 / 3, 4)
    assert s1["top_helper_comment_count_on_post"] == 2

    monthly = result.monthly.set_index(["subreddit", "month"])
    askreddit = monthly.loc[("askreddit", "2022-10")]
    assert askreddit["community_type"] == "general"
    assert askreddit["submissions"] == 1
    assert askreddit["reply_rate"] == 1.0
    assert askreddit["unanswered_rate"] == 0.0
    assert askreddit["op_followup_rate"] == 1.0
    assert askreddit["mean_post_max_depth"] == 3.0

    depression = monthly.loc[("depression", "2022-10")]
    assert depression["community_type"] == "health"
    assert depression["submissions"] == 2
    assert depression["reply_rate"] == 0.5
    assert depression["unanswered_rate"] == 0.5
    assert round(float(depression["median_first_reply_latency_hours"]), 4) == round(5 / 60, 4)


def test_responsiveness_cache_invalidates_after_input_change(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir, cache_dir = _make_processed(tmp_path)

    run_responsiveness_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
    )

    assert load_responsiveness_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is not None

    mutated_comments = COMMENTS + [{
        "id": "c5",
        "subreddit": "depression",
        "author": "helper3",
        "parent_id": "t3_s3",
        "link_id": "t3_s3",
        "created_utc": 1664600900,
    }]
    write_zst_jsonl(comments_path, mutated_comments)

    assert load_responsiveness_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is None


def test_run_responsiveness_analysis_partitioned_matches_default(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir, cache_dir = _make_processed(tmp_path)
    config = normalize_thread_prep_config(2, cache_dir=tmp_path / "thread-prep-cache")
    assert config is not None

    default_result = run_responsiveness_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
    )
    partitioned_result = run_responsiveness_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tmp_path / "tables-partitioned",
        figures_dir=tmp_path / "figures-partitioned",
        cache_dir=tmp_path / "cache-partitioned",
        thread_prep=config,
    )

    pd.testing.assert_frame_equal(partitioned_result.posts, default_result.posts)
    pd.testing.assert_frame_equal(partitioned_result.monthly, default_result.monthly)
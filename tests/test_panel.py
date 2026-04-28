"""Tests for src.panel — monthly subreddit analysis panel."""

from __future__ import annotations

from pathlib import Path

from src.panel import build_monthly_panel, ensure_monthly_panel, load_panel_cache
from tests.conftest import write_zst_jsonl


SUBMISSIONS = [
    {
        "id": "s_general_oct",
        "subreddit": "askreddit",
        "author": "op_general",
        "title": "General question",
        "selftext": "Need answers soon",
        "score": 5,
        "created_utc": 1664600000,
    },
    {
        "id": "s_health_nov",
        "subreddit": "depression",
        "author": "op_health",
        "title": "Need support",
        "selftext": "I feel anxious today",
        "score": 3,
        "created_utc": 1667300000,
    },
    {
        "id": "s_health_removed_nov",
        "subreddit": "depression",
        "author": "[deleted]",
        "title": "Removed thread",
        "selftext": "[removed]",
        "score": 0,
        "created_utc": 1667300600,
    },
]

COMMENTS = [
    {
        "id": "c_general_1",
        "subreddit": "askreddit",
        "author": "helper_a",
        "body": "hello there",
        "score": 10,
        "parent_id": "t3_s_general_oct",
        "link_id": "t3_s_general_oct",
        "created_utc": 1664600100,
    },
    {
        "id": "c_general_2",
        "subreddit": "askreddit",
        "author": "helper_b",
        "body": "[removed]",
        "score": 0,
        "parent_id": "t1_c_general_1",
        "link_id": "t3_s_general_oct",
        "created_utc": 1664600200,
    },
    {
        "id": "c_health_1",
        "subreddit": "depression",
        "author": "helper_h1",
        "body": "I am sorry",
        "score": 5,
        "parent_id": "t3_s_health_nov",
        "link_id": "t3_s_health_nov",
        "created_utc": 1667300100,
    },
    {
        "id": "c_health_2",
        "subreddit": "depression",
        "author": "helper_h1",
        "body": "take care",
        "score": 2,
        "parent_id": "t1_c_health_1",
        "link_id": "t3_s_health_nov",
        "created_utc": 1667300200,
    },
    {
        "id": "c_health_3",
        "subreddit": "depression",
        "author": "helper_h2",
        "body": "doctor visit soon",
        "score": 1,
        "parent_id": "t3_s_health_nov",
        "link_id": "t3_s_health_nov",
        "created_utc": 1667300300,
    },
    {
        "id": "c_orphan_dec",
        "subreddit": "askreddit",
        "author": "helper_c",
        "body": "lonely comment",
        "score": 1,
        "parent_id": "t3_missing_submission",
        "link_id": "t3_missing_submission",
        "created_utc": 1670000000,
    },
]


def _make_processed(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"
    comments_path = processed_dir / "filter-comments-test.jsonl.zst"
    submissions_path = processed_dir / "filter-submissions-test.jsonl.zst"
    write_zst_jsonl(comments_path, COMMENTS)
    write_zst_jsonl(submissions_path, SUBMISSIONS)
    return comments_path, submissions_path, tables_dir


def test_build_monthly_panel_aggregates_expected_metrics(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir = _make_processed(tmp_path)

    result = build_monthly_panel(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        cache_dir=tables_dir,
    )

    rows = {(row.subreddit, row.month): row for row in result.rows}
    assert set(rows) == {
        ("askreddit", "2022-10"),
        ("askreddit", "2022-11"),
        ("askreddit", "2022-12"),
        ("depression", "2022-10"),
        ("depression", "2022-11"),
        ("depression", "2022-12"),
    }

    empty_general_nov = rows[("askreddit", "2022-11")]
    assert empty_general_nov.comments == 0
    assert empty_general_nov.submissions == 0
    assert empty_general_nov.community_type == "general"
    assert empty_general_nov.comments_per_submission == 0.0

    general_oct = rows[("askreddit", "2022-10")]
    assert general_oct.community_type == "general"
    assert general_oct.comments == 2
    assert general_oct.submissions == 1
    assert general_oct.comments_per_submission == 2.0
    assert general_oct.unique_comment_authors == 2
    assert general_oct.unique_submission_authors == 1
    assert general_oct.mean_comment_length == float(len("hello there"))
    assert general_oct.median_comment_length == float(len("hello there"))
    assert general_oct.deleted_removed_comment_share == 0.5
    assert general_oct.mean_submission_title_length == float(len("General question"))
    assert general_oct.mean_submission_selftext_length == float(len("Need answers soon"))
    assert general_oct.mean_depth == 1.5
    assert general_oct.threading_ratio == 0.5
    assert general_oct.max_depth == 2
    assert general_oct.top1_share == 0.5
    assert general_oct.post_genai == 0
    assert general_oct.months_since_genai == -1

    health_nov = rows[("depression", "2022-11")]
    assert health_nov.community_type == "health"
    assert health_nov.comments == 3
    assert health_nov.submissions == 2
    assert health_nov.comments_per_submission == 1.5
    assert health_nov.unique_comment_authors == 2
    assert health_nov.unique_submission_authors == 1
    assert health_nov.mean_comment_length == (
        len("I am sorry") + len("take care") + len("doctor visit soon")
    ) / 3
    assert health_nov.median_comment_length == float(len("I am sorry"))
    assert health_nov.deleted_removed_submission_share == 0.5
    assert health_nov.mean_submission_title_length == float(len("Need support"))
    assert health_nov.mean_submission_selftext_length == float(len("I feel anxious today"))
    assert round(health_nov.mean_depth, 4) == round(4 / 3, 4)
    assert round(health_nov.threading_ratio, 4) == round(1 / 3, 4)
    assert health_nov.max_depth == 2
    assert round(health_nov.top1_share, 4) == round(2 / 3, 4)
    assert health_nov.post_genai == 1
    assert health_nov.months_since_genai == 0


def test_build_monthly_panel_safe_division_for_missing_submissions(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir = _make_processed(tmp_path)

    result = build_monthly_panel(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        cache_dir=tables_dir,
    )

    orphan_row = next(
        row for row in result.rows if row.subreddit == "askreddit" and row.month == "2022-12"
    )
    assert orphan_row.submissions == 0
    assert orphan_row.comments == 1
    assert orphan_row.comments_per_submission == 0.0


def test_panel_cache_fingerprint_invalidates_after_input_change(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir = _make_processed(tmp_path)

    csv_path, metadata_path, metadata = ensure_monthly_panel(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
    )

    assert csv_path.exists()
    assert metadata_path.exists()
    assert load_panel_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
    ) == metadata

    mutated_comments = COMMENTS + [{
        "id": "c_mutated",
        "subreddit": "askreddit",
        "author": "helper_d",
        "body": "new evidence",
        "score": 7,
        "parent_id": "t3_s_general_oct",
        "link_id": "t3_s_general_oct",
        "created_utc": 1664600300,
    }]
    write_zst_jsonl(comments_path, mutated_comments)

    assert load_panel_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
    ) is None
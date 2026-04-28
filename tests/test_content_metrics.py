"""Tests for src.content_metrics — monthly text proxy metrics and caching."""

from __future__ import annotations

from pathlib import Path

from src.content_metrics import load_content_metrics_cache, run_content_metrics_analysis
from tests.conftest import write_zst_jsonl


SUBMISSIONS = [
    {
        "id": "s1",
        "subreddit": "askreddit",
        "title": "How do I fix this?",
        "selftext": "Anyone have experience with quick repairs?",
        "created_utc": 1667300000,
    },
    {
        "id": "s2",
        "subreddit": "depression",
        "title": "My doctor said to take care",
        "selftext": "I have symptoms and hope for support.",
        "created_utc": 1667300100,
    },
]

COMMENTS = [
    {
        "id": "c1",
        "subreddit": "askreddit",
        "body": "Why not ask anyone else?",
        "created_utc": 1667300200,
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
    },
    {
        "id": "c2",
        "subreddit": "depression",
        "body": "I am sorry. Take care. Ask your doctor.",
        "created_utc": 1667300300,
        "parent_id": "t3_s2",
        "link_id": "t3_s2",
    },
]


def _make_processed(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    processed_dir = tmp_path / "processed"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    comments_path = processed_dir / "filter-comments-test.jsonl.zst"
    submissions_path = processed_dir / "filter-submissions-test.jsonl.zst"
    write_zst_jsonl(comments_path, COMMENTS)
    write_zst_jsonl(submissions_path, SUBMISSIONS)
    return comments_path, submissions_path, tables_dir, figures_dir


def test_run_content_metrics_analysis_computes_expected_shares(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir = _make_processed(tmp_path)

    result = run_content_metrics_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    monthly = result.monthly.set_index(["subreddit", "month"])
    askreddit = monthly.loc[("askreddit", "2022-11")]
    assert askreddit["community_type"] == "general"
    assert askreddit["comment_question_share"] == 1.0
    assert askreddit["submission_question_share"] == 1.0

    depression = monthly.loc[("depression", "2022-11")]
    assert depression["community_type"] == "health"
    assert depression["comment_support_share"] == 1.0
    assert depression["comment_medical_share"] == 1.0
    assert depression["submission_experience_share"] == 1.0


def test_content_metrics_cache_invalidates_after_input_change(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir = _make_processed(tmp_path)

    run_content_metrics_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    assert load_content_metrics_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is not None

    mutated_comments = COMMENTS + [{
        "id": "c3",
        "subreddit": "askreddit",
        "body": "Support and hope matter too.",
        "created_utc": 1667300400,
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
    }]
    write_zst_jsonl(comments_path, mutated_comments)

    assert load_content_metrics_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is None
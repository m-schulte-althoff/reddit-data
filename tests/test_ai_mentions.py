"""Tests for src.ai_mentions — monthly AI mention counts and cache behavior."""

from __future__ import annotations

from pathlib import Path

from src.ai_mentions import load_ai_mentions_cache, run_ai_mentions_analysis
from tests.conftest import write_zst_jsonl


SUBMISSIONS = [
    {
        "id": "s1",
        "subreddit": "askreddit",
        "author": "op1",
        "title": "ChatGPT changed homework",
        "selftext": "OpenAI tools are everywhere",
        "created_utc": 1667300000,
    },
    {
        "id": "s2",
        "subreddit": "depression",
        "author": "op2",
        "title": "Need support",
        "selftext": "No AI mention here",
        "created_utc": 1667300100,
    },
]

COMMENTS = [
    {
        "id": "c1",
        "subreddit": "askreddit",
        "author": "u1",
        "body": "Copilot is fine, but I still ask people.",
        "created_utc": 1667300200,
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
    },
    {
        "id": "c2",
        "subreddit": "depression",
        "author": "u2",
        "body": "No mention.",
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


def test_run_ai_mentions_analysis_counts_mentions_and_shares(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir = _make_processed(tmp_path)

    result = run_ai_mentions_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    monthly = result.monthly.set_index(["subreddit", "month"])
    askreddit = monthly.loc[("askreddit", "2022-11")]
    assert askreddit["community_type"] == "general"
    assert askreddit["ai_mention_comments"] == 1
    assert askreddit["ai_mention_submissions"] == 1
    assert askreddit["ai_mention_comment_share"] == 1.0
    assert askreddit["ai_mention_submission_share"] == 1.0

    depression = monthly.loc[("depression", "2022-11")]
    assert depression["community_type"] == "health"
    assert depression["ai_mention_comments"] == 0
    assert depression["ai_mention_submissions"] == 0


def test_ai_mentions_cache_invalidates_after_input_change(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir = _make_processed(tmp_path)

    run_ai_mentions_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )

    assert load_ai_mentions_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is not None

    mutated_comments = COMMENTS + [{
        "id": "c3",
        "subreddit": "depression",
        "author": "u3",
        "body": "ChatGPT helped me phrase this.",
        "created_utc": 1667300400,
        "parent_id": "t3_s2",
        "link_id": "t3_s2",
    }]
    write_zst_jsonl(comments_path, mutated_comments)

    assert load_ai_mentions_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is None
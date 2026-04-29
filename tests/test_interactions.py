"""Tests for src.interactions — bond-vs-identity interaction metrics."""

from __future__ import annotations

from pathlib import Path

from src.interactions import load_interactions_cache, run_interactions_analysis
from src.thread_prep import normalize_thread_prep_config
from tests.conftest import write_zst_jsonl


SUBMISSIONS = [
    {
        "id": "g0",
        "subreddit": "askreddit",
        "author": "opg0",
        "title": "September thread",
        "created_utc": 1662000000,
    },
    {
        "id": "g1",
        "subreddit": "askreddit",
        "author": "opg1",
        "title": "October thread",
        "created_utc": 1664600000,
    },
    {
        "id": "g2",
        "subreddit": "askreddit",
        "author": "opg2",
        "title": "November thread",
        "created_utc": 1667300000,
    },
    {
        "id": "h1",
        "subreddit": "depression",
        "author": "oph1",
        "title": "Health October thread",
        "created_utc": 1664600000,
    },
]

COMMENTS = [
    {
        "id": "g0c1",
        "subreddit": "askreddit",
        "author": "dana",
        "parent_id": "t3_g0",
        "link_id": "t3_g0",
        "created_utc": 1662000300,
    },
    {
        "id": "g1c1",
        "subreddit": "askreddit",
        "author": "alice",
        "parent_id": "t3_g1",
        "link_id": "t3_g1",
        "created_utc": 1664600300,
    },
    {
        "id": "g1c2",
        "subreddit": "askreddit",
        "author": "bob",
        "parent_id": "t3_g1",
        "link_id": "t3_g1",
        "created_utc": 1664600400,
    },
    {
        "id": "g2c1",
        "subreddit": "askreddit",
        "author": "alice",
        "parent_id": "t3_g2",
        "link_id": "t3_g2",
        "created_utc": 1667300300,
    },
    {
        "id": "g2c2",
        "subreddit": "askreddit",
        "author": "opg2",
        "parent_id": "t1_g2c1",
        "link_id": "t3_g2",
        "created_utc": 1667300400,
    },
    {
        "id": "g2c3",
        "subreddit": "askreddit",
        "author": "alice",
        "parent_id": "t1_g2c2",
        "link_id": "t3_g2",
        "created_utc": 1667300500,
    },
    {
        "id": "g2c4",
        "subreddit": "askreddit",
        "author": "dana",
        "parent_id": "t3_g2",
        "link_id": "t3_g2",
        "created_utc": 1667300600,
    },
    {
        "id": "g2c5",
        "subreddit": "askreddit",
        "author": "charlie",
        "parent_id": "t3_g2",
        "link_id": "t3_g2",
        "created_utc": 1667300700,
    },
    {
        "id": "g2c6",
        "subreddit": "askreddit",
        "author": "alice",
        "parent_id": "t1_g2c3",
        "link_id": "t3_g2",
        "created_utc": 1667300800,
    },
    {
        "id": "h1c1",
        "subreddit": "depression",
        "author": "helper1",
        "parent_id": "t3_h1",
        "link_id": "t3_h1",
        "created_utc": 1664600100,
    },
    {
        "id": "h1c2",
        "subreddit": "depression",
        "author": "oph1",
        "parent_id": "t1_h1c1",
        "link_id": "t3_h1",
        "created_utc": 1664600200,
    },
    {
        "id": "h1c3",
        "subreddit": "depression",
        "author": "helper2",
        "parent_id": "t1_h1c2",
        "link_id": "t3_h1",
        "created_utc": 1664600300,
    },
    {
        "id": "h1c4",
        "subreddit": "depression",
        "author": "helper3",
        "parent_id": "t3_h1",
        "link_id": "t3_h1",
        "created_utc": 1664600400,
    },
    {
        "id": "h1c5",
        "subreddit": "depression",
        "author": "helper4",
        "parent_id": "t3_h1",
        "link_id": "t3_h1",
        "created_utc": 1664600500,
    },
    {
        "id": "h1c6",
        "subreddit": "depression",
        "author": "helper5",
        "parent_id": "t3_h1",
        "link_id": "t3_h1",
        "created_utc": 1664600600,
    },
    {
        "id": "h1c7",
        "subreddit": "depression",
        "author": "helper6",
        "parent_id": "t3_h1",
        "link_id": "t3_h1",
        "created_utc": 1664600700,
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


def test_run_interactions_analysis_computes_expected_metrics(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir, cache_dir = _make_processed(tmp_path)

    result = run_interactions_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
    )

    monthly = result.monthly.set_index(["subreddit", "month"])

    askreddit_nov = monthly.loc[("askreddit", "2022-11")]
    assert askreddit_nov["community_type"] == "general"
    assert askreddit_nov["unique_authors"] == 4
    assert round(float(askreddit_nov["new_author_share"]), 4) == 0.5
    assert round(float(askreddit_nov["returning_author_share"]), 4) == 0.25
    assert round(float(askreddit_nov["repeat_author_share"]), 4) == 0.25
    assert askreddit_nov["op_return_rate"] == 1.0
    assert round(float(askreddit_nov["reciprocal_dyad_share"]), 4) == round(1 / 3, 4)
    assert round(float(askreddit_nov["repeat_dyad_share"]), 4) == 0.6
    assert askreddit_nov["single_commenter_thread_share"] == 0.0
    assert askreddit_nov["multi_actor_thread_share"] == 1.0
    assert askreddit_nov["focused_thread_share"] == 1.0
    assert askreddit_nov["distributed_thread_share"] == 0.0

    depression_oct = monthly.loc[("depression", "2022-10")]
    assert depression_oct["community_type"] == "health"
    assert depression_oct["unique_authors"] == 7
    assert depression_oct["new_author_share"] == 1.0
    assert depression_oct["returning_author_share"] == 0.0
    assert depression_oct["repeat_author_share"] == 0.0
    assert depression_oct["op_return_rate"] == 1.0
    assert round(float(depression_oct["reciprocal_dyad_share"]), 4) == round(1 / 6, 4)
    assert round(float(depression_oct["repeat_dyad_share"]), 4) == round(2 / 7, 4)
    assert depression_oct["single_commenter_thread_share"] == 0.0
    assert depression_oct["multi_actor_thread_share"] == 1.0
    assert depression_oct["focused_thread_share"] == 0.0
    assert depression_oct["distributed_thread_share"] == 1.0

    assert float(askreddit_nov["bond_index"]) > float(depression_oct["bond_index"])
    assert float(depression_oct["identity_index"]) > float(askreddit_nov["identity_index"])


def test_interactions_cache_invalidates_after_input_change(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir, cache_dir = _make_processed(tmp_path)

    run_interactions_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
    )

    assert load_interactions_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is not None

    mutated_comments = COMMENTS + [{
        "id": "extra",
        "subreddit": "askreddit",
        "author": "helperx",
        "parent_id": "t3_g2",
        "link_id": "t3_g2",
        "created_utc": 1667300900,
    }]
    write_zst_jsonl(comments_path, mutated_comments)

    assert load_interactions_cache(
        [comments_path],
        [submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    ) is None


def test_run_interactions_analysis_partitioned_matches_default(tmp_path: Path) -> None:
    comments_path, submissions_path, tables_dir, figures_dir, cache_dir = _make_processed(tmp_path)
    config = normalize_thread_prep_config(2, cache_dir=tmp_path / "thread-prep-cache")
    assert config is not None

    default_result = run_interactions_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        cache_dir=cache_dir,
    )
    partitioned_result = run_interactions_analysis(
        comment_paths=[comments_path],
        submission_paths=[submissions_path],
        tables_dir=tmp_path / "tables-partitioned",
        figures_dir=tmp_path / "figures-partitioned",
        cache_dir=tmp_path / "cache-partitioned",
        thread_prep=config,
    )

    assert partitioned_result.monthly.to_dict("records") == default_result.monthly.to_dict("records")
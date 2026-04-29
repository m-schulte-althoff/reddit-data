"""Tests for src.thread_prep — submission-hash partition prep."""

from __future__ import annotations

from pathlib import Path

from src.thread_prep import normalize_thread_prep_config, prepare_thread_partitions
from tests.conftest import read_zst_jsonl, write_zst_jsonl


SUBMISSIONS = [
    {
        "id": "s1",
        "subreddit": "askreddit",
        "author": "op1",
        "created_utc": 1664600000,
    },
    {
        "id": "s2",
        "subreddit": "depression",
        "author": "op2",
        "created_utc": 1664600100,
    },
]

COMMENTS = [
    {
        "id": "c1",
        "subreddit": "askreddit",
        "author": "helper1",
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
        "created_utc": 1664600200,
    },
    {
        "id": "c2",
        "subreddit": "askreddit",
        "author": "helper2",
        "parent_id": "t1_c1",
        "link_id": "t3_s1",
        "created_utc": 1664600300,
    },
    {
        "id": "c3",
        "subreddit": "depression",
        "author": "helper3",
        "parent_id": "t3_s2",
        "link_id": "t3_s2",
        "created_utc": 1664600400,
    },
]


def _make_processed(tmp_path: Path) -> tuple[Path, Path]:
    processed_dir = tmp_path / "processed"
    comments_path = processed_dir / "filter-comments-test.jsonl.zst"
    submissions_path = processed_dir / "filter-submissions-test.jsonl.zst"
    write_zst_jsonl(comments_path, COMMENTS)
    write_zst_jsonl(submissions_path, SUBMISSIONS)
    return comments_path, submissions_path


def test_normalize_thread_prep_config_disables_single_partition() -> None:
    assert normalize_thread_prep_config(None) is None
    assert normalize_thread_prep_config(1) is None


def test_prepare_thread_partitions_keeps_each_submission_in_one_shard(tmp_path: Path) -> None:
    comments_path, submissions_path = _make_processed(tmp_path)
    config = normalize_thread_prep_config(2, cache_dir=tmp_path / "cache")
    assert config is not None

    artifacts = prepare_thread_partitions(
        [comments_path],
        [submissions_path],
        config=config,
    )

    submission_shards: dict[str, int] = {}
    for index, shard_path in enumerate(artifacts.submission_partitions):
        for record in read_zst_jsonl(shard_path):
            submission_shards[str(record["id"])] = index

    assert submission_shards == {"s1": submission_shards["s1"], "s2": submission_shards["s2"]}
    assert len({submission_shards["s1"], submission_shards["s2"]}) >= 1

    comment_shards: dict[str, int] = {}
    for index, shard_path in enumerate(artifacts.comment_partitions):
        for record in read_zst_jsonl(shard_path):
            comment_shards[str(record["id"])] = index

    assert comment_shards["c1"] == submission_shards["s1"]
    assert comment_shards["c2"] == submission_shards["s1"]
    assert comment_shards["c3"] == submission_shards["s2"]


def test_prepare_thread_partitions_reuses_valid_cache(tmp_path: Path) -> None:
    comments_path, submissions_path = _make_processed(tmp_path)
    config = normalize_thread_prep_config(2, cache_dir=tmp_path / "cache")
    assert config is not None

    first = prepare_thread_partitions([comments_path], [submissions_path], config=config)
    second = prepare_thread_partitions([comments_path], [submissions_path], config=config)

    assert first.manifest_path == second.manifest_path
    assert first.metadata == second.metadata
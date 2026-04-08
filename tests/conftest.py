"""Shared test fixtures — tiny .zst samples for fast tests."""

from __future__ import annotations

import io
from pathlib import Path

import orjson
import zstandard as zstd
import pytest


SAMPLE_COMMENTS: list[dict] = [
    {
        "id": "c1",
        "body": "hello world",
        "subreddit": "askreddit",
        "subreddit_name_prefixed": "r/askreddit",
        "author": "user1",
        "score": 10,
        "created_utc": 1654100000,  # 2022-06-01T19:33:20Z
    },
    {
        "id": "c2",
        "body": "test comment",
        "subreddit": "science",
        "subreddit_name_prefixed": "r/science",
        "author": "user2",
        "score": 5,
        "created_utc": 1654200000,  # 2022-06-02T23:20:00Z
    },
    {
        "id": "c3",
        "body": "off-topic",
        "subreddit": "funny",
        "subreddit_name_prefixed": "r/funny",
        "author": "user3",
        "score": 100,
        "created_utc": 1654300000,  # 2022-06-04T03:06:40Z
    },
    {
        "id": "c4",
        "body": "old comment",
        "subreddit": "askreddit",
        "subreddit_name_prefixed": "r/askreddit",
        "author": "user4",
        "score": 1,
        "created_utc": 1609459200,  # 2021-01-01 (outside default window)
    },
    {
        "id": "c5",
        "body": "mental health",
        "subreddit": "Depression",
        "subreddit_name_prefixed": "r/Depression",
        "author": "user5",
        "score": 3,
        "created_utc": 1654400000,  # 2022-06-05T06:53:20Z
    },
]

SAMPLE_SUBMISSIONS: list[dict] = [
    {
        "id": "s1",
        "title": "Test post",
        "subreddit": "askreddit",
        "subreddit_name_prefixed": "r/askreddit",
        "author": "user1",
        "score": 50,
        "created_utc": 1654100000,
    },
    {
        "id": "s2",
        "title": "Science post",
        "subreddit": "science",
        "subreddit_name_prefixed": "r/science",
        "author": "user2",
        "score": 200,
        "created_utc": 1654200000,
    },
    {
        "id": "s3",
        "title": "Funny meme",
        "subreddit": "funny",
        "subreddit_name_prefixed": "r/funny",
        "author": "user3",
        "score": 1000,
        "created_utc": 1654300000,
    },
]


def write_zst_jsonl(path: Path, records: list[dict]) -> None:
    """Compress *records* as JSONL into a .zst file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    buf = b"\n".join(orjson.dumps(r) for r in records) + b"\n"
    with path.open("wb") as f:
        with cctx.stream_writer(f) as zout:
            zout.write(buf)


def read_zst_jsonl(path: Path) -> list[dict]:
    """Decompress a .zst JSONL file and return parsed records."""
    dctx = zstd.ZstdDecompressor()
    records: list[dict] = []
    with path.open("rb") as fin:
        with dctx.stream_reader(fin) as zin:
            buf = io.BufferedReader(zin)
            for line in buf:
                line = line.strip()
                if line:
                    records.append(orjson.loads(line))
    return records


@pytest.fixture()
def raw_tree(tmp_path: Path) -> Path:
    """Create a minimal raw data tree with comments + submissions .zst files.

    Returns the base directory (tmp_path) so tests can construct paths like
    ``base / "data" / "raw" / "reddit" / "comments" / "RC_2022-06.zst"``.
    """
    comments_dir = tmp_path / "data" / "raw" / "reddit" / "comments"
    submissions_dir = tmp_path / "data" / "raw" / "reddit" / "submissions"
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    write_zst_jsonl(comments_dir / "RC_2022-06.zst", SAMPLE_COMMENTS)
    write_zst_jsonl(submissions_dir / "RS_2022-06.zst", SAMPLE_SUBMISSIONS)
    return tmp_path


@pytest.fixture()
def subreddit_list_file(tmp_path: Path) -> Path:
    """Write a subreddit list file and return its path."""
    p = tmp_path / "subreddit-list.txt"
    p.write_text(
        'subreddits = [\n'
        '    "r/askreddit",\n'
        '    "r/science",\n'
        '    "r/Depression",\n'
        ']\n',
        encoding="utf-8",
    )
    return p

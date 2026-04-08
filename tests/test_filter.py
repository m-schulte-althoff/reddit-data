"""Tests for src/filter.py — subreddit filtering with resume."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import read_zst_jsonl, write_zst_jsonl
from src.filter import (
    _discover_raw_paths,
    _load_progress,
    _save_progress,
    filter_by_subreddit,
    load_subreddit_list,
)


# ── load_subreddit_list ─────────────────────────────────────────────────────


def test_load_subreddit_list(subreddit_list_file: Path) -> None:
    subs = load_subreddit_list(subreddit_list_file)
    assert "r/askreddit" in subs
    assert "r/science" in subs
    assert "r/depression" in subs  # lowercased
    assert len(subs) == 3


def test_load_subreddit_list_bad_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("nothing here", encoding="utf-8")
    try:
        load_subreddit_list(p)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── _discover_raw_paths ─────────────────────────────────────────────────────


def test_discover_raw_paths_all(raw_tree: Path) -> None:
    raw_dir = raw_tree / "data" / "raw"
    paths = _discover_raw_paths("comments", raw_dir=raw_dir)
    assert len(paths) == 1
    assert paths[0].name == "RC_2022-06.zst"


def test_discover_raw_paths_month_filter(raw_tree: Path) -> None:
    raw_dir = raw_tree / "data" / "raw"
    paths = _discover_raw_paths("comments", raw_dir=raw_dir, months=[(2022, 6)])
    assert len(paths) == 1

    paths = _discover_raw_paths("comments", raw_dir=raw_dir, months=[(2099, 1)])
    assert len(paths) == 0


# ── Progress tracking ───────────────────────────────────────────────────────


def test_progress_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "output.jsonl.zst"
    _save_progress(out, {"completed_files": ["a.zst"], "rows_written": 42})
    loaded = _load_progress(out)
    assert loaded["completed_files"] == ["a.zst"]
    assert loaded["rows_written"] == 42


def test_load_progress_missing(tmp_path: Path) -> None:
    out = tmp_path / "nonexistent.jsonl.zst"
    loaded = _load_progress(out)
    assert loaded["completed_files"] == []
    assert loaded["rows_written"] == 0


# ── filter_by_subreddit ─────────────────────────────────────────────────────


def test_filter_comments_by_subreddit(raw_tree: Path) -> None:
    """Filter comments keeping only askreddit + science."""
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    subreddits = {"r/askreddit", "r/science"}
    result = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=False,
    )

    assert result["rows_written"] == 3  # c1 (askreddit), c2 (science), c4 (askreddit old)
    output_path = Path(result["output_file"])
    assert output_path.exists()

    records = read_zst_jsonl(output_path)
    ids = {r["id"] for r in records}
    assert ids == {"c1", "c2", "c4"}


def test_filter_comments_by_subreddit_with_time_window(raw_tree: Path) -> None:
    """Filter with both subreddit and time window."""
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    subreddits = {"r/askreddit", "r/science"}
    result = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_tw",
        raw_dir=raw_dir,
        output_dir=out_dir,
        start_epoch=1654000000,   # 2022-06-01 approx
        end_epoch=1654250000,     # cuts off c4 (old) already excluded, c2 still in
        resume=False,
    )

    output_path = Path(result["output_file"])
    records = read_zst_jsonl(output_path)
    ids = {r["id"] for r in records}
    # c1 ts=1654100000 ✓, c2 ts=1654200000 ✓, c4 ts=1609459200 ✗ (too old)
    assert ids == {"c1", "c2"}


def test_filter_submissions(raw_tree: Path) -> None:
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    subreddits = {"r/askreddit"}
    result = filter_by_subreddit(
        kind="submissions",
        subreddits=subreddits,
        output_tag="test_sub",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=False,
    )

    assert result["rows_written"] == 1  # s1 only
    records = read_zst_jsonl(Path(result["output_file"]))
    assert records[0]["id"] == "s1"


def test_filter_excludes_non_matching(raw_tree: Path) -> None:
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    subreddits = {"r/nonexistent"}
    result = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_empty",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=False,
    )
    assert result["rows_written"] == 0


def test_filter_case_insensitive(raw_tree: Path) -> None:
    """Subreddit matching should be case-insensitive."""
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    # r/Depression in data, r/depression in filter set (lowercased)
    subreddits = {"r/depression"}
    result = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_case",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=False,
    )

    assert result["rows_written"] == 1
    records = read_zst_jsonl(Path(result["output_file"]))
    assert records[0]["id"] == "c5"


# ── Resume ───────────────────────────────────────────────────────────────────


def test_filter_resume_skips_completed(raw_tree: Path) -> None:
    """A second run with resume=True should not duplicate records."""
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    subreddits = {"r/askreddit", "r/science"}

    # First run
    r1 = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_resume",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=True,
    )

    # Second run — should skip
    r2 = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_resume",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=True,
    )

    assert r2["rows_read"] == 0  # nothing new processed
    assert r2.get("resumed") is True

    # Output file should still have the same records
    records = read_zst_jsonl(Path(r1["output_file"]))
    assert len(records) == 3


def test_filter_resume_adds_new_file(raw_tree: Path) -> None:
    """Resume picks up a newly added .zst file."""
    raw_dir = raw_tree / "data" / "raw"
    out_dir = raw_tree / "data" / "processed"

    subreddits = {"r/askreddit"}

    # First run — processes RC_2022-06.zst
    r1 = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_resume2",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=True,
    )
    assert r1["rows_written"] == 2  # c1 + c4

    # Add a second file
    extra_comments = [
        {
            "id": "c_extra",
            "body": "new",
            "subreddit": "askreddit",
            "subreddit_name_prefixed": "r/askreddit",
            "author": "user99",
            "score": 7,
            "created_utc": 1656700000,
        },
    ]
    comments_dir = raw_dir / "reddit" / "comments"
    write_zst_jsonl(comments_dir / "RC_2022-07.zst", extra_comments)

    # Resume — should only process the new file
    r2 = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_resume2",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=True,
    )
    assert len(r2["per_file"]) == 1
    assert r2["per_file"][0]["input_file"] == "RC_2022-07.zst"
    assert r2["rows_written"] == 3  # 2 + 1

    # All records accessible from output
    records = read_zst_jsonl(Path(r1["output_file"]))
    ids = {r["id"] for r in records}
    assert ids == {"c1", "c4", "c_extra"}


def test_filter_no_files(tmp_path: Path) -> None:
    """Empty raw directory should return zero stats without error."""
    raw_dir = tmp_path / "data" / "raw"
    (raw_dir / "reddit" / "comments").mkdir(parents=True)
    out_dir = tmp_path / "data" / "processed"

    result = filter_by_subreddit(
        kind="comments",
        subreddits={"r/askreddit"},
        output_tag="test_none",
        raw_dir=raw_dir,
        output_dir=out_dir,
    )
    assert result["rows_read"] == 0
    assert result["rows_written"] == 0

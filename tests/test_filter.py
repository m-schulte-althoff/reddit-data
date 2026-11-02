"""Tests for src/filter.py — subreddit filtering with resume."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

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


def test_filter_submissions_excludes_crosspost_parent_only_match(tmp_path: Path) -> None:
    """Nested crosspost parent subreddit fields must not admit off-list posts."""
    raw_dir = tmp_path / "data" / "raw"
    out_dir = tmp_path / "data" / "processed"
    submissions_dir = raw_dir / "reddit" / "submissions"
    records = [
        {
            "id": "x1",
            "title": "Crosspost from science",
            "crosspost_parent_list": [
                {
                    "subreddit": "science",
                    "subreddit_name_prefixed": "r/science",
                    "title": "Original science post",
                },
            ],
            "subreddit": "alphafold",
            "subreddit_name_prefixed": "r/alphafold",
            "author": "user1",
            "created_utc": 1654100000,
        },
        {
            "id": "sci1",
            "title": "Direct science post",
            "subreddit": "science",
            "subreddit_name_prefixed": "r/science",
            "author": "user2",
            "created_utc": 1654200000,
        },
    ]
    write_zst_jsonl(submissions_dir / "RS_2022-06.zst", records)

    result = filter_by_subreddit(
        kind="submissions",
        subreddits={"r/science"},
        output_tag="test_crosspost",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=False,
    )

    assert result["rows_written"] == 1
    output_records = read_zst_jsonl(Path(result["output_file"]))
    assert [record["id"] for record in output_records] == ["sci1"]


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


def test_filter_json_escaped_slash(tmp_path: Path) -> None:
    """Real Arctic Shift data encodes '/' as '\\/' in JSON — filter must handle."""
    raw_dir = tmp_path / "data" / "raw"
    out_dir = tmp_path / "data" / "processed"
    comments_dir = raw_dir / "reddit" / "comments"

    # Simulate real data: orjson doesn't escape '/', so write raw bytes manually.
    import zstandard as zstd

    escaped_records = (
        b'{"id":"e1","subreddit":"askreddit","subreddit_name_prefixed":"r\\/askreddit","created_utc":1654100000,"score":1}\n'
        b'{"id":"e2","subreddit":"science","subreddit_name_prefixed":"r\\/science","created_utc":1654200000,"score":2}\n'
        b'{"id":"e3","subreddit":"funny","subreddit_name_prefixed":"r\\/funny","created_utc":1654300000,"score":3}\n'
    )
    comments_dir.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with (comments_dir / "RC_2022-06.zst").open("wb") as f:
        with cctx.stream_writer(f) as zout:
            zout.write(escaped_records)

    subreddits = {"r/askreddit", "r/science"}
    result = filter_by_subreddit(
        kind="comments",
        subreddits=subreddits,
        output_tag="test_escaped",
        raw_dir=raw_dir,
        output_dir=out_dir,
        resume=False,
    )

    assert result["rows_written"] == 2
    records = read_zst_jsonl(Path(result["output_file"]))
    ids = {r["id"] for r in records}
    assert ids == {"e1", "e2"}


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


@pytest.mark.parametrize("interrupted_month", ["2022-06", "2022-07"])
def test_filter_resume_discards_interrupted_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted_month: str,
) -> None:
    """Retrying a partial first or later input must write each record only once."""
    from src import filter as module

    raw_dir, output_dir = tmp_path / "raw", tmp_path / "processed"
    records = {
        month: [{"id": month + name, "subreddit": "science"} for name in ("a", "b")]
        for month in ("2022-06", "2022-07")
    }
    for month, rows in records.items():
        write_zst_jsonl(raw_dir / "reddit/comments" / f"RC_{month}.zst", rows)
    original = module._filter_file

    def interrupted(path, zout, *args):
        if interrupted_month in path.name:
            zout.write(orjson.dumps(records[interrupted_month][0]) + b"\n")
            raise RuntimeError("interrupted")
        return original(path, zout, *args)

    monkeypatch.setattr(module, "_filter_file", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        filter_by_subreddit("comments", {"r/science"}, raw_dir=raw_dir, output_dir=output_dir)
    monkeypatch.setattr(module, "_filter_file", original)
    result = filter_by_subreddit("comments", {"r/science"}, raw_dir=raw_dir, output_dir=output_dir)
    output = Path(result["output_file"])
    assert [r["id"] for r in read_zst_jsonl(output)] == [
        "2022-06a", "2022-06b", "2022-07a", "2022-07b",
    ]
    assert result["rows_written"] == 4
    assert _load_progress(output)["output_bytes"] == output.stat().st_size
    before = output.read_bytes()
    filter_by_subreddit("comments", {"r/science"}, raw_dir=raw_dir, output_dir=output_dir)
    assert output.read_bytes() == before


@pytest.mark.parametrize("problem", ["legacy", "missing", "short", "orphan"])
def test_filter_resume_rejects_unsafe_checkpoint(raw_tree: Path, problem: str) -> None:
    """An invalid checkpoint must fail before changing any surviving data."""
    raw_dir, output_dir = raw_tree / "data/raw", raw_tree / "data/processed"
    result = filter_by_subreddit("comments", {"r/science"}, raw_dir=raw_dir, output_dir=output_dir)
    output = Path(result["output_file"])
    progress = _load_progress(output)
    if problem == "legacy":
        del progress["output_bytes"]
        _save_progress(output, progress)
    elif problem == "missing":
        output.unlink()
    elif problem == "short":
        output.write_bytes(output.read_bytes()[:-1])
    else:
        output.with_suffix(".zst.progress.json").unlink()
    before = output.read_bytes() if output.exists() else None
    with pytest.raises((ValueError, FileNotFoundError)):
        filter_by_subreddit("comments", {"r/science"}, raw_dir=raw_dir, output_dir=output_dir)
    assert (output.read_bytes() if output.exists() else None) == before


def test_save_progress_preserves_symlink_and_previous_checkpoint_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic updates preserve the sidecar symlink and the previous valid JSON."""
    from src import filter as module

    target = tmp_path / "checkpoint.json"
    target.write_text('{"rows_written": 1}')
    output = tmp_path / "data.zst"
    sidecar = output.with_suffix(".zst.progress.json")
    sidecar.symlink_to(target)
    _save_progress(output, {"rows_written": 2})
    assert sidecar.is_symlink()
    previous = target.read_bytes()

    def failed_replace(*args):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(module.os, "fsync", failed_replace)
    with pytest.raises(OSError, match="checkpoint failure"):
        _save_progress(output, {"rows_written": 3})
    assert target.read_bytes() == previous
    assert sidecar.is_symlink()
    assert not list(tmp_path.glob("*.tmp"))

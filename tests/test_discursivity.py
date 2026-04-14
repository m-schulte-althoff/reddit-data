"""Tests for src/discursivity.py — comment depth and threading analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import write_zst_jsonl
from src.discursivity import (
    DepthBucket,
    DiscursivityResult,
    _cascade_resolve,
    _PendingComment,
    compute_discursivity,
)


# ── Sample data with a known comment tree ────────────────────────────────────
#
# Submission s1 (r/askreddit, 2022-06)
# ├── c1  depth 1  (parent_id: t3_s1)
# │   ├── c2  depth 2  (parent_id: t1_c1)
# │   │   └── c4  depth 3  (parent_id: t1_c2)
# │   └── c5  depth 2  (parent_id: t1_c1)
# └── c3  depth 1  (parent_id: t3_s1)
#
# Submission s2 (r/science, 2022-07)
# └── c6  depth 1  (parent_id: t3_s2)
#     └── c7  depth 2  (parent_id: t1_c6)

SAMPLE_SUBMISSIONS = [
    {
        "id": "s1",
        "subreddit": "askreddit",
        "subreddit_name_prefixed": "r/askreddit",
        "created_utc": 1654100000,   # 2022-06
    },
    {
        "id": "s2",
        "subreddit": "science",
        "subreddit_name_prefixed": "r/science",
        "created_utc": 1656700000,   # 2022-07
    },
]

SAMPLE_COMMENTS = [
    # ── Thread under s1 ──
    {
        "id": "c1",
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
        "subreddit": "askreddit",
        "created_utc": 1654100100,   # 2022-06
    },
    {
        "id": "c2",
        "parent_id": "t1_c1",
        "link_id": "t3_s1",
        "subreddit": "askreddit",
        "created_utc": 1654100200,
    },
    {
        "id": "c3",
        "parent_id": "t3_s1",
        "link_id": "t3_s1",
        "subreddit": "askreddit",
        "created_utc": 1654100300,
    },
    {
        "id": "c4",
        "parent_id": "t1_c2",
        "link_id": "t3_s1",
        "subreddit": "askreddit",
        "created_utc": 1654100400,
    },
    {
        "id": "c5",
        "parent_id": "t1_c1",
        "link_id": "t3_s1",
        "subreddit": "askreddit",
        "created_utc": 1654100500,
    },
    # ── Thread under s2 ──
    {
        "id": "c6",
        "parent_id": "t3_s2",
        "link_id": "t3_s2",
        "subreddit": "science",
        "created_utc": 1656700100,   # 2022-07
    },
    {
        "id": "c7",
        "parent_id": "t1_c6",
        "link_id": "t3_s2",
        "subreddit": "science",
        "created_utc": 1656700200,
    },
]


def _make_processed(tmp_path: Path) -> tuple[Path, Path]:
    """Write sample .zst files and return (comments_path, submissions_path)."""
    d = tmp_path / "processed"
    d.mkdir(parents=True, exist_ok=True)
    cp = d / "filter-comments-test.jsonl.zst"
    sp = d / "filter-submissions-test.jsonl.zst"
    write_zst_jsonl(cp, SAMPLE_COMMENTS)
    write_zst_jsonl(sp, SAMPLE_SUBMISSIONS)
    return cp, sp


# ── DepthBucket unit tests ───────────────────────────────────────────────────


def test_depth_bucket_empty() -> None:
    b = DepthBucket()
    assert b.count == 0
    assert b.mean_depth == 0.0
    assert b.threading_ratio == 0.0


def test_depth_bucket_add() -> None:
    b = DepthBucket()
    b.add(1)
    b.add(2)
    b.add(2)
    b.add(3)
    assert b.count == 4
    assert b.max_depth == 3
    assert b.mean_depth == 2.0  # (1+2+2+3)/4
    assert b.threading_ratio == 0.75  # 3 of 4 at depth >= 2


# ── Cascade resolve unit test ────────────────────────────────────────────────


def test_cascade_resolve() -> None:
    result = DiscursivityResult()
    depth_map: dict[str, int] = {"parent": 1}
    pending: dict[str, list[_PendingComment]] = {
        "parent": [_PendingComment("child1", "sub", "2022-06")],
        "child1": [_PendingComment("grandchild", "sub", "2022-06")],
    }
    _cascade_resolve("parent", 1, depth_map, pending, result)

    assert depth_map["child1"] == 2
    assert depth_map["grandchild"] == 3
    assert result.resolved_comments == 2
    assert len(pending) == 0


# ── Integration: compute_discursivity ────────────────────────────────────────


def test_compute_discursivity_basic(tmp_path: Path) -> None:
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    assert result.total_comments == 7
    assert result.resolved_comments == 7
    assert result.unresolved_comments == 0


def test_compute_discursivity_submission_counts(tmp_path: Path) -> None:
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    assert result.submission_counts[("askreddit", "2022-06")] == 1
    assert result.submission_counts[("science", "2022-07")] == 1


def test_compute_discursivity_depths_askreddit(tmp_path: Path) -> None:
    """Check depth distribution for askreddit in 2022-06."""
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    bucket = result.buckets[("askreddit", "2022-06")]
    assert bucket.count == 5
    # c1=1, c2=2, c3=1, c4=3, c5=2
    assert bucket.depth_histogram[1] == 2  # c1, c3
    assert bucket.depth_histogram[2] == 2  # c2, c5
    assert bucket.depth_histogram[3] == 1  # c4
    assert bucket.max_depth == 3
    assert bucket.mean_depth == pytest.approx(1.8)  # (1+2+1+3+2)/5
    assert bucket.threading_ratio == pytest.approx(0.6)  # 3/5


def test_compute_discursivity_depths_science(tmp_path: Path) -> None:
    """Check depth distribution for science in 2022-07."""
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    bucket = result.buckets[("science", "2022-07")]
    assert bucket.count == 2
    assert bucket.depth_histogram[1] == 1  # c6
    assert bucket.depth_histogram[2] == 1  # c7
    assert bucket.max_depth == 2
    assert bucket.threading_ratio == pytest.approx(0.5)


def test_compute_discursivity_sorted_subreddits(tmp_path: Path) -> None:
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    subs = result.sorted_subreddits()
    assert subs[0] == "askreddit"  # 5 comments
    assert subs[1] == "science"    # 2 comments


def test_compute_discursivity_unresolved(tmp_path: Path) -> None:
    """Comment with parent outside the filtered set → unresolved."""
    d = tmp_path / "processed"
    d.mkdir(parents=True, exist_ok=True)
    orphan = [
        {"id": "x1", "parent_id": "t1_missing", "subreddit": "test", "created_utc": 1654100000},
        {"id": "x2", "parent_id": "t3_s1", "subreddit": "test", "created_utc": 1654100000},
    ]
    cp = d / "filter-comments-orphan.jsonl.zst"
    write_zst_jsonl(cp, orphan)
    sp = d / "filter-submissions-empty.jsonl.zst"
    write_zst_jsonl(sp, [])

    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])
    assert result.resolved_comments == 1  # x2
    assert result.unresolved_comments == 1  # x1


def test_compute_discursivity_out_of_order(tmp_path: Path) -> None:
    """Child appears before parent in the file — cascade resolves it."""
    d = tmp_path / "processed"
    d.mkdir(parents=True, exist_ok=True)
    records = [
        # child first
        {"id": "b", "parent_id": "t1_a", "subreddit": "test", "created_utc": 1654100000},
        # parent second
        {"id": "a", "parent_id": "t3_s1", "subreddit": "test", "created_utc": 1654100000},
    ]
    cp = d / "filter-comments-ooo.jsonl.zst"
    write_zst_jsonl(cp, records)
    sp = d / "filter-submissions-empty.jsonl.zst"
    write_zst_jsonl(sp, [])

    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])
    assert result.resolved_comments == 2
    assert result.unresolved_comments == 0
    # a=depth 1, b=depth 2
    bucket = result.buckets[("test", "2022-06")]
    assert bucket.depth_histogram[1] == 1
    assert bucket.depth_histogram[2] == 1


# ── View tests ───────────────────────────────────────────────────────────────


def test_write_discursivity_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from views import write_discursivity_csv

    monkeypatch.setattr("src.config.TABLES_DIR", tmp_path)
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    out = write_discursivity_csv(result, "test-discursivity.csv")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "subreddit" in text
    assert "mean_depth" in text
    assert "threading_ratio" in text
    assert "askreddit" in text
    assert "ALL" in text


def test_plot_discursivity_mean_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import plot_discursivity_mean_depth

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    out = plot_discursivity_mean_depth(result, "test-depth.svg")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_discursivity_threading_ratio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import plot_discursivity_threading_ratio

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    cp, sp = _make_processed(tmp_path)
    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])

    out = plot_discursivity_threading_ratio(result, "test-threading.svg")
    assert out.exists()
    assert out.stat().st_size > 0


def test_compute_discursivity_integer_ids(tmp_path: Path) -> None:
    """parent_id / id may be integers in some records — must not crash."""
    d = tmp_path / "processed"
    d.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": 100, "parent_id": "t3_s1", "subreddit": "test", "created_utc": 1654100000},
        {"id": "c2", "parent_id": 999, "subreddit": "test", "created_utc": 1654100000},
    ]
    cp = d / "filter-comments-intid.jsonl.zst"
    write_zst_jsonl(cp, records)
    sp = d / "filter-submissions-empty.jsonl.zst"
    write_zst_jsonl(sp, [])

    result = compute_discursivity(comment_paths=[cp], submission_paths=[sp])
    assert result.total_comments == 2
    # First comment has t3_ parent → depth 1 → resolved
    assert result.resolved_comments >= 1

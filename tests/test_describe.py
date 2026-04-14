"""Tests for src/describe.py and describe-related views."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import write_zst_jsonl
from src.describe import DescribeResult, _epoch_to_month, describe_filtered


# ── Sample data spanning multiple months + subreddits ────────────────────────

FILTERED_RECORDS: list[dict] = [
    # r/askreddit — June 2022
    {"id": "c1", "subreddit": "askreddit", "created_utc": 1654100000, "score": 10},
    {"id": "c2", "subreddit": "askreddit", "created_utc": 1654200000, "score": 5},
    # r/science — June 2022
    {"id": "c3", "subreddit": "science", "created_utc": 1654300000, "score": 20},
    # r/askreddit — July 2022
    {"id": "c4", "subreddit": "askreddit", "created_utc": 1656700000, "score": 3},
    # r/Depression — July 2022
    {"id": "c5", "subreddit": "Depression", "created_utc": 1656800000, "score": 1},
    # r/science — August 2022
    {"id": "c6", "subreddit": "science", "created_utc": 1659400000, "score": 15},
    # r/askreddit — August 2022
    {"id": "c7", "subreddit": "askreddit", "created_utc": 1659500000, "score": 8},
    {"id": "c8", "subreddit": "askreddit", "created_utc": 1659600000, "score": 12},
]


# ── Unit tests: helpers ──────────────────────────────────────────────────────


def test_epoch_to_month() -> None:
    assert _epoch_to_month(1654100000) == "2022-06"
    assert _epoch_to_month(1656700000) == "2022-07"
    assert _epoch_to_month(1659400000) == "2022-08"


# ── Unit tests: DescribeResult ───────────────────────────────────────────────


def test_describe_result_empty() -> None:
    r = DescribeResult(kind="comments")
    assert r.total_records == 0
    assert r.sorted_months() == []
    assert r.sorted_subreddits() == []
    t_min, t_max = r.time_range()
    assert t_min is None and t_max is None


def test_describe_result_update_ts() -> None:
    r = DescribeResult(kind="comments")
    r.update_ts(200)
    r.update_ts(100)
    r.update_ts(300)
    assert r.min_ts == 100
    assert r.max_ts == 300


# ── Integration: describe_filtered with sample .zst ──────────────────────────


def _make_filtered_zst(tmp_path: Path) -> Path:
    """Create a tiny filtered .zst file and return its path."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    p = processed / "filter-comments-test.jsonl.zst"
    write_zst_jsonl(p, FILTERED_RECORDS)
    return p


def test_describe_filtered_counts(tmp_path: Path) -> None:
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    assert result.total_records == 8
    assert result.kind == "comments"
    assert result.parse_errors == 0


def test_describe_filtered_subreddit_counts(tmp_path: Path) -> None:
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    assert result.subreddit_counts["askreddit"] == 5
    assert result.subreddit_counts["science"] == 2
    assert result.subreddit_counts["Depression"] == 1
    assert len(result.subreddit_counts) == 3


def test_describe_filtered_monthly_counts(tmp_path: Path) -> None:
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    assert result.monthly_counts["2022-06"] == 3
    assert result.monthly_counts["2022-07"] == 2
    assert result.monthly_counts["2022-08"] == 3
    assert result.sorted_months() == ["2022-06", "2022-07", "2022-08"]


def test_describe_filtered_subreddit_monthly(tmp_path: Path) -> None:
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    assert result.subreddit_monthly_counts[("askreddit", "2022-06")] == 2
    assert result.subreddit_monthly_counts[("askreddit", "2022-07")] == 1
    assert result.subreddit_monthly_counts[("askreddit", "2022-08")] == 2
    assert result.subreddit_monthly_counts[("science", "2022-06")] == 1
    assert result.subreddit_monthly_counts[("science", "2022-08")] == 1
    assert result.subreddit_monthly_counts[("Depression", "2022-07")] == 1


def test_describe_filtered_time_range(tmp_path: Path) -> None:
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    t_min, t_max = result.time_range()
    assert t_min == "2022-06-01"
    assert t_max == "2022-08-04"


def test_describe_filtered_empty(tmp_path: Path) -> None:
    """No matching files → empty result."""
    result = describe_filtered("comments", input_paths=[])
    assert result.total_records == 0


# ── View tests ───────────────────────────────────────────────────────────────


def test_write_describe_summary_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from views import write_describe_summary_csv

    monkeypatch.setattr("src.config.TABLES_DIR", tmp_path)
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    out = write_describe_summary_csv(result, "test-summary.csv")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "total_records" in text
    assert "8" in text  # total count
    assert "subreddit:askreddit" in text


def test_write_describe_monthly_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from views import write_describe_monthly_csv

    monkeypatch.setattr("src.config.TABLES_DIR", tmp_path)
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    out = write_describe_monthly_csv(result, "test-monthly.csv")
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    header = lines[0]
    assert "subreddit" in header
    assert "2022-06" in header
    assert "total" in header
    # 3 subreddits + ALL row + header = 5 lines
    assert len(lines) == 5


def test_write_describe_monthly_csv_top_n(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import write_describe_monthly_csv

    monkeypatch.setattr("src.config.TABLES_DIR", tmp_path)
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    out = write_describe_monthly_csv(result, "test-monthly-top2.csv", top_n=2)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    assert lines[1].startswith("askreddit,")
    assert lines[2].startswith("science,")
    assert lines[3].startswith("ALL,")


def test_plot_describe_trend_aggregated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from views import plot_describe_trend_aggregated

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    out = plot_describe_trend_aggregated(result, "test-agg.svg")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_describe_trend_per_subreddit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from views import plot_describe_trend_per_subreddit

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    out = plot_describe_trend_per_subreddit(result, "test-per-sub.svg", top_n=3)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_describe_trend_all_subreddits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import plot_describe_trend_per_subreddit

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    zst_path = _make_filtered_zst(tmp_path)
    result = describe_filtered("comments", input_paths=[zst_path])

    out = plot_describe_trend_per_subreddit(result, "test-per-sub-all.svg", top_n=None)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_describe_trend_caps_many_subreddits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When subreddit count exceeds _MAX_TREND_LINES, the plot is capped."""
    from views import plot_describe_trend_per_subreddit, _MAX_TREND_LINES

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)

    # Build a DescribeResult with more subreddits than the cap.
    result = DescribeResult(kind="comments")
    n_subs = _MAX_TREND_LINES + 20
    for i in range(n_subs):
        sub = f"sub_{i:04d}"
        result.subreddit_counts[sub] = n_subs - i  # descending rank
        result.monthly_counts["2022-06"] += 1
        result.subreddit_monthly_counts[(sub, "2022-06")] = n_subs - i
    result.total_records = sum(result.subreddit_counts.values())
    result.min_ts = 1654100000
    result.max_ts = 1654100000

    out = plot_describe_trend_per_subreddit(result, "test-capped.svg", top_n=None)
    assert out.exists()
    # SVG file should be reasonably small (not bloated).
    assert out.stat().st_size < 500_000

"""Tests for src/analysis.py — stats accumulator."""

from src.analysis import StreamStats


def test_stream_stats_empty() -> None:
    s = StreamStats()
    d = s.as_dict()
    assert d["total_rows_scanned"] == 0
    assert d["rows_in_window"] == 0
    assert d["score_mean"] is None


def test_stream_stats_update_ts() -> None:
    s = StreamStats()
    s.update_ts(100)
    s.update_ts(200)
    s.update_ts(50)
    assert s.min_created_utc == 50
    assert s.max_created_utc == 200


def test_stream_stats_update_score() -> None:
    s = StreamStats()
    s.update_score(10)
    s.update_score(20)
    s.update_score(30)
    d = s.as_dict()
    assert d["score_count"] == 3
    assert d["score_mean"] == 20.0
    assert d["score_min"] == 10
    assert d["score_max"] == 30

"""Tests for src/analysis.py — stats accumulator and helpers."""

from src.analysis import StreamStats, _cache_key, _merge_reservoirs


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


def test_cache_key_deterministic() -> None:
    k1 = _cache_key("comments", 500, 42, None)
    k2 = _cache_key("comments", 500, 42, None)
    assert k1 == k2
    assert len(k1) == 12


def test_cache_key_varies_with_params() -> None:
    k_all = _cache_key("comments", 500, 42, None)
    k_month = _cache_key("comments", 500, 42, [(2022, 10)])
    k_seed = _cache_key("comments", 500, 99, None)
    assert k_all != k_month
    assert k_all != k_seed


def test_merge_reservoirs_empty() -> None:
    assert _merge_reservoirs([], n=10, seed=42) == []


def test_merge_reservoirs_single() -> None:
    records = [{"id": i} for i in range(5)]
    merged = _merge_reservoirs([(5, records)], n=5, seed=42)
    assert len(merged) == 5


def test_merge_reservoirs_trims_to_n() -> None:
    r1 = [{"id": i} for i in range(10)]
    r2 = [{"id": i + 10} for i in range(10)]
    merged = _merge_reservoirs([(100, r1), (100, r2)], n=5, seed=42)
    assert len(merged) == 5

"""Tests for src/arctic_shift.py — helper functions."""

from src.arctic_shift import MonthRef, iter_months, build_target_paths, build_magnet_uri


def test_iter_months_single() -> None:
    months = iter_months((2022, 11), (2022, 11))
    assert months == [MonthRef(2022, 11)]


def test_iter_months_cross_year() -> None:
    months = iter_months((2022, 11), (2023, 2))
    assert len(months) == 4
    assert months[0] == MonthRef(2022, 11)
    assert months[-1] == MonthRef(2023, 2)


def test_build_target_paths_keys() -> None:
    paths = build_target_paths()
    assert "comments" in paths
    assert "submissions" in paths
    assert len(paths["comments"]) == len(paths["submissions"])


def test_build_target_paths_format() -> None:
    paths = build_target_paths()
    for p in paths["comments"]:
        assert p.startswith("reddit/comments/RC_")
        assert p.endswith(".zst")
    for p in paths["submissions"]:
        assert p.startswith("reddit/submissions/RS_")
        assert p.endswith(".zst")


def test_build_magnet_uri_contains_hash() -> None:
    uri = build_magnet_uri("abc123", ["udp://tracker.example.com:1234"])
    assert "xt=urn:btih:abc123" in uri
    assert "tr=udp://tracker.example.com:1234" in uri

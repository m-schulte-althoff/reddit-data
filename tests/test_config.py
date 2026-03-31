"""Tests for src/config.py — derived month ranges."""

from src.config import END_MONTH, START_MONTH, START_TS, END_EXCLUSIVE_TS


def test_start_month_derived_from_start_ts() -> None:
    assert START_MONTH == (START_TS.year, START_TS.month)


def test_end_month_contains_last_relevant_second() -> None:
    # The last second in-window is END_EXCLUSIVE_TS - 1s.
    from datetime import timedelta

    last_second = END_EXCLUSIVE_TS - timedelta(seconds=1)
    assert END_MONTH == (last_second.year, last_second.month)


def test_month_range_reasonable() -> None:
    sy, sm = START_MONTH
    ey, em = END_MONTH
    assert (sy, sm) <= (ey, em), "START_MONTH must not be after END_MONTH"

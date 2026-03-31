"""Tests for src/config.py — derived month ranges."""

from src.config import END_MONTH, START_MONTH, START_TS, END_EXCLUSIVE_TS


def test_start_month_derived_from_start_ts() -> None:
    assert START_MONTH == (START_TS.year, START_TS.month)


def test_end_month_contains_last_relevant_second() -> None:
    # END_EXCLUSIVE_TS is 2022-12-30 00:00:00 UTC.
    # The last second in-window is 2022-12-29 23:59:59 -> December.
    assert END_MONTH[0] == (END_EXCLUSIVE_TS.year)
    assert END_MONTH[1] == 12


def test_month_range_reasonable() -> None:
    sy, sm = START_MONTH
    ey, em = END_MONTH
    assert (sy, sm) <= (ey, em), "START_MONTH must not be after END_MONTH"

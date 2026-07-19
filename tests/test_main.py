"""CLI tests for main.py."""

from __future__ import annotations

import sys
from pathlib import Path

import main
from src.thread_prep import ThreadPrepConfig


def test_cmd_panel_passes_thread_prep_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, int] = {}

    def fake_ensure_monthly_panel(*, thread_prep: ThreadPrepConfig | None = None, **_: object):
        captured["partitions"] = 0 if thread_prep is None else thread_prep.partitions
        return tmp_path / "panel.csv", tmp_path / "panel-metadata.json", {"n_rows": 0, "n_subreddits": 0}

    from src import panel as panel_module

    monkeypatch.setattr(panel_module, "ensure_monthly_panel", fake_ensure_monthly_panel)
    monkeypatch.setattr(sys, "argv", ["main.py", "panel", "--thread-prep-partitions", "8"])

    assert main.cmd_panel() == 0
    assert captured["partitions"] == 8


def test_cmd_filter_subreddit_passes_kind(monkeypatch) -> None:
    captured: dict[str, tuple[str, ...]] = {}

    def fake_filter_all(**kwargs: object) -> dict[str, dict[str, object]]:
        kinds = kwargs["kinds"]
        assert isinstance(kinds, tuple)
        captured["kinds"] = kinds
        return {"submissions": {"rows_written": 1, "output_file": "out.zst"}}

    from src import filter as filter_module

    monkeypatch.setattr(filter_module, "filter_all", fake_filter_all)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "filter-subreddit", "--kind", "submissions", "--tag", "repair"],
    )

    assert main.cmd_filter_subreddit() == 0
    assert captured["kinds"] == ("submissions",)


def test_cmd_support_capacity_passes_window_months(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, int] = {}

    class Result:
        table_paths = {"monthly": tmp_path / "support-capacity-monthly.csv"}

    def fake_run_support_capacity_analysis(*, window_months: int) -> Result:
        captured["window_months"] = window_months
        return Result()

    from src import support_capacity as support_capacity_module

    monkeypatch.setattr(
        support_capacity_module,
        "run_support_capacity_analysis",
        fake_run_support_capacity_analysis,
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "support-capacity", "--window-months", "24"])

    assert main.cmd_support_capacity() == 0
    assert captured["window_months"] == 24

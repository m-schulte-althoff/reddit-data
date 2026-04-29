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
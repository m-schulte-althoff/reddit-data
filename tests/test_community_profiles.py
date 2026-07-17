"""Tests for exploratory health-community growth profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.community_profiles import run_community_profile_analysis


def test_community_profiles_link_pre_period_structure(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index in range(3):
        subreddit = f"health_{index}"
        for month, post, comments in [
            ("2022-10", False, 100 + index),
            ("2022-11", True, 120 + index * 20),
        ]:
            rows.append({
                "subreddit": subreddit,
                "month": month,
                "community_type": "health",
                "comments": comments,
                "submissions": 10,
                "threading_ratio": 0.2,
                "gini": 0.4,
                "top5_share": 0.1,
            })
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False)

    result = run_community_profile_analysis(panel_path=panel_path, tables_dir=tmp_path)

    assert len(result.profiles) == 3
    assert set(result.profiles["analysis_label"]) == {"exploratory_descriptive"}
    assert result.table_paths["summary"].exists()
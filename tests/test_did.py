"""Tests for src.did — DiD and event-study estimation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.did import (
    ModelSpec,
    estimate_twfe_did,
    prepare_analysis_frame,
    run_did_analysis,
    run_event_study,
)


def _make_synthetic_panel(*, unbalanced: bool = False) -> pd.DataFrame:
    months = [
        "2022-08",
        "2022-09",
        "2022-10",
        "2022-11",
        "2022-12",
        "2023-01",
        "2023-02",
        "2023-03",
        "2023-04",
    ]
    rows: list[dict[str, object]] = []

    for idx in range(4):
        subreddit = f"general_{idx}"
        for month_idx, month in enumerate(months):
            post = month >= "2022-11"
            comments = 100 + idx * 2 + month_idx
            submissions = 20 + idx + (month_idx % 3)
            if post:
                comments -= 40 - (month_idx % 2)
                submissions -= 8 - (idx % 2)
            rows.append({
                "subreddit": subreddit,
                "month": month,
                "community_type": "general",
                "comments": comments,
                "submissions": submissions,
                "comments_per_submission": comments / submissions,
            })

    for idx in range(4):
        subreddit = f"health_{idx}"
        for month_idx, month in enumerate(months):
            post = month >= "2022-11"
            comments = 100 + idx * 2 + month_idx
            submissions = 20 + idx + (month_idx % 3)
            if post:
                comments -= 10 - ((idx + month_idx) % 3)
                submissions -= 2 - (month_idx % 2)
            rows.append({
                "subreddit": subreddit,
                "month": month,
                "community_type": "health",
                "comments": comments,
                "submissions": submissions,
                "comments_per_submission": comments / submissions,
            })

    frame = pd.DataFrame(rows)
    if unbalanced:
        frame = frame.loc[
            ~((frame["subreddit"] == "general_0") & (frame["month"] == "2023-04"))
        ].reset_index(drop=True)
    return frame


def test_estimate_twfe_did_recovers_positive_health_post_effect() -> None:
    panel = _make_synthetic_panel()

    result = estimate_twfe_did(panel, "comments", spec=ModelSpec(model="unweighted"))

    assert result["estimate"] == pytest.approx(0.32, abs=0.2)
    assert result["std_error"] > 0
    assert result["n_subreddits"] == 8


def test_event_study_omits_reference_period_and_clusters_run() -> None:
    panel = _make_synthetic_panel()

    event_study = run_event_study(panel, "comments")

    assert -1 not in event_study["event_time"].tolist()
    assert event_study["std_error"].notna().all()
    assert (event_study.loc[event_study["event_time"] >= 0, "estimate"] > 0).all()


def test_balanced_panel_filter_removes_unbalanced_subreddit() -> None:
    panel = _make_synthetic_panel(unbalanced=True)

    unbalanced = prepare_analysis_frame(panel, "comments")
    balanced = prepare_analysis_frame(panel, "comments", balanced_only=True)

    assert len(balanced) < len(unbalanced)
    assert "general_0" not in set(balanced["subreddit"])


def test_run_did_analysis_is_deterministic(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    panel = _make_synthetic_panel()
    panel.to_csv(panel_path, index=False)

    first = run_did_analysis(panel_path=panel_path, tables_dir=tables_dir, figures_dir=figures_dir)
    second = run_did_analysis(panel_path=panel_path, tables_dir=tables_dir, figures_dir=figures_dir)

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.event_studies["comments"], second.event_studies["comments"])
    assert first.table_paths["summary"].exists()
    assert first.figure_paths["event_comments"].exists()
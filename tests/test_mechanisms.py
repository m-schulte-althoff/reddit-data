"""Tests for src.mechanisms — moderation models on merged panel data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mechanisms import build_moderation_panel, estimate_moderation_model, run_mechanisms_analysis


def _make_panel() -> pd.DataFrame:
    months = ["2022-09", "2022-10", "2022-11", "2022-12"]
    rows: list[dict[str, object]] = []
    specs = [
        ("general_hi_1", "general", 0.8, 60),
        ("general_hi_2", "general", 0.8, 62),
        ("general_lo_1", "general", 0.2, 60),
        ("general_lo_2", "general", 0.2, 61),
        ("health_hi_1", "health", 0.8, 95),
        ("health_hi_2", "health", 0.8, 94),
        ("health_lo_1", "health", 0.2, 70),
        ("health_lo_2", "health", 0.2, 69),
    ]
    for subreddit, community_type, top5_share, post_comments in specs:
        for month in months:
            post = month >= "2022-11"
            comments = 100 if not post else post_comments
            submissions = 20 if not post else 18
            rows.append({
                "subreddit": subreddit,
                "month": month,
                "community_type": community_type,
                "comments": comments,
                "submissions": submissions,
                "comments_per_submission": comments / submissions,
                "top5_share": top5_share,
                "top1_share": top5_share / 2,
                "gini": top5_share / 2,
                "hhi": top5_share / 2,
                "pct1_share": top5_share / 2,
                "threading_ratio": 0.6 if top5_share > 0.5 else 0.2,
                "mean_depth": 2.0 if top5_share > 0.5 else 1.0,
            })
    return pd.DataFrame(rows)


def _make_responsiveness() -> pd.DataFrame:
    months = ["2022-09", "2022-10", "2022-11", "2022-12"]
    rows: list[dict[str, object]] = []
    specs = [
        ("general_hi_1", "general", 0.2),
        ("general_hi_2", "general", 0.2),
        ("general_lo_1", "general", 0.1),
        ("general_lo_2", "general", 0.1),
        ("health_hi_1", "health", 0.9),
        ("health_hi_2", "health", 0.9),
        ("health_lo_1", "health", 0.4),
        ("health_lo_2", "health", 0.4),
    ]
    for subreddit, community_type, reply_rate in specs:
        for month in months:
            rows.append({
                "subreddit": subreddit,
                "month": month,
                "community_type": community_type,
                "submissions": 1,
                "reply_rate": reply_rate,
                "unanswered_rate": 1 - reply_rate,
                "median_first_reply_latency_hours": 1.0 - reply_rate / 2,
                "p25_first_reply_latency_hours": 0.5,
                "p75_first_reply_latency_hours": 1.5,
                "mean_unique_commenters": 2.0,
                "median_unique_commenters": 2.0,
                "mean_non_op_commenters": 1.0,
                "op_followup_rate": reply_rate / 2,
                "mean_op_followup_comments": reply_rate,
                "mean_post_threading_ratio": reply_rate / 2,
                "mean_post_max_depth": 2.0,
            })
    return pd.DataFrame(rows)


def test_estimate_moderation_model_returns_positive_top5_interaction() -> None:
    merged = build_moderation_panel(_make_panel(), _make_responsiveness())

    result = estimate_moderation_model(merged, "comments", "pre_top5_share")

    assert result["estimate"] > 0
    assert result["n_subreddits"] == 8


def test_run_mechanisms_analysis_writes_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    _make_panel().to_csv(panel_path, index=False)

    from src import mechanisms as mechanisms_module

    original_runner = mechanisms_module.run_responsiveness_analysis
    try:
        mechanisms_module.run_responsiveness_analysis = lambda: type(
            "FakeResponsiveness",
            (),
            {"monthly": _make_responsiveness()},
        )()
        artifacts = run_mechanisms_analysis(
            panel_path=panel_path,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )
    finally:
        mechanisms_module.run_responsiveness_analysis = original_runner

    assert not artifacts.summary.empty
    assert artifacts.table_paths["summary"].exists()
    assert artifacts.figure_paths["coefficients"].exists()
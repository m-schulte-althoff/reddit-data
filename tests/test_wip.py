"""Tests for src.wip — orchestration and key-result summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import wip as wip_module


def test_run_wip_suite_writes_summary_files(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    panel = pd.DataFrame([
        {
            "subreddit": "askreddit",
            "month": "2022-10",
            "community_type": "general",
            "comments": 10,
            "submissions": 2,
            "top5_share": 0.5,
            "gini": 0.4,
        },
        {
            "subreddit": "depression",
            "month": "2022-11",
            "community_type": "health",
            "comments": 12,
            "submissions": 3,
            "top5_share": 0.3,
            "gini": 0.2,
        },
    ])
    panel_path = tables_dir / "community-monthly-panel.csv"
    tables_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False)

    original_ensure_panel = wip_module.ensure_monthly_panel
    original_did = wip_module.run_did_analysis
    original_resp = wip_module.run_responsiveness_analysis
    original_mech = wip_module.run_mechanisms_analysis
    original_ai = wip_module.run_ai_mentions_analysis
    original_content = wip_module.run_content_metrics_analysis
    original_interactions = wip_module.run_interactions_analysis
    try:
        wip_module.ensure_monthly_panel = lambda **_: (panel_path, panel_path, {"n_rows": 2})
        wip_module.run_did_analysis = lambda **_: type("Did", (), {"summary": pd.DataFrame([
            {"outcome": "comments", "model": "unweighted", "estimate": 0.1, "std_error": 0.01, "p_value": 0.05},
            {"outcome": "submissions", "model": "unweighted", "estimate": 0.2, "std_error": 0.02, "p_value": 0.04},
            {"outcome": "comments_per_submission", "model": "unweighted", "estimate": 0.3, "std_error": 0.03, "p_value": 0.03},
            {"outcome": "comments", "model": "weighted_pre_activity", "estimate": 0.1, "std_error": 0.01, "p_value": 0.05},
            {"outcome": "comments", "model": "balanced_panel", "estimate": 0.1, "std_error": 0.01, "p_value": 0.05},
            {"outcome": "comments", "model": "exclude_top2_general", "estimate": 0.1, "std_error": 0.01, "p_value": 0.05},
            {"outcome": "comments", "model": "winsorized_1_99", "estimate": 0.1, "std_error": 0.01, "p_value": 0.05},
            {"outcome": "comments", "model": "cutoff_2023_03", "estimate": 0.1, "std_error": 0.01, "p_value": 0.05},
        ])})()
        wip_module.run_responsiveness_analysis = lambda **_: type("Resp", (), {"monthly": pd.DataFrame([
            {"subreddit": "askreddit", "month": "2022-10", "community_type": "general", "reply_rate": 0.5, "unanswered_rate": 0.5, "op_followup_rate": 0.1},
            {"subreddit": "depression", "month": "2022-11", "community_type": "health", "reply_rate": 0.7, "unanswered_rate": 0.3, "op_followup_rate": 0.2},
        ])})()
        wip_module.run_mechanisms_analysis = lambda **_: type("Mech", (), {"summary": pd.DataFrame([
            {"model_type": "triple_interaction", "outcome": "comments", "moderator": "pre_top5_share", "estimate": 0.2, "p_value": 0.01},
        ])})()
        wip_module.run_ai_mentions_analysis = lambda **_: type("Ai", (), {"monthly": pd.DataFrame([
            {"subreddit": "askreddit", "month": "2022-10", "community_type": "general", "ai_mention_comment_share": 0.1, "ai_mention_submission_share": 0.2},
            {"subreddit": "depression", "month": "2022-11", "community_type": "health", "ai_mention_comment_share": 0.05, "ai_mention_submission_share": 0.06},
        ])})()
        wip_module.run_content_metrics_analysis = lambda **_: type("Content", (), {"monthly": pd.DataFrame([
            {"subreddit": "askreddit", "month": "2022-10", "community_type": "general", "comment_question_share": 0.5, "submission_experience_share": 0.1, "comment_support_share": 0.1},
            {"subreddit": "depression", "month": "2022-11", "community_type": "health", "comment_question_share": 0.2, "submission_experience_share": 0.6, "comment_support_share": 0.7},
        ])})()
        wip_module.run_interactions_analysis = lambda **_: type("Interactions", (), {"monthly": pd.DataFrame([
            {"subreddit": "askreddit", "month": "2022-10", "community_type": "general", "bond_index": -0.2, "identity_index": 0.4},
            {"subreddit": "depression", "month": "2022-11", "community_type": "health", "bond_index": 0.5, "identity_index": 0.1},
        ])})()

        artifacts = wip_module.run_wip_suite(tables_dir=tables_dir, figures_dir=figures_dir)
    finally:
        wip_module.ensure_monthly_panel = original_ensure_panel
        wip_module.run_did_analysis = original_did
        wip_module.run_responsiveness_analysis = original_resp
        wip_module.run_mechanisms_analysis = original_mech
        wip_module.run_ai_mentions_analysis = original_ai
        wip_module.run_content_metrics_analysis = original_content
        wip_module.run_interactions_analysis = original_interactions

    assert not artifacts.summary_table.empty
    assert artifacts.csv_path.exists()
    assert artifacts.markdown_path.exists()
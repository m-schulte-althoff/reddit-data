"""Tests for interaction metrics as analytic outcomes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.interaction_outcomes import run_interaction_outcomes_analysis


def test_interaction_outcomes_excludes_history_warmup_rows(tmp_path: Path) -> None:
    months = [
        "2022-06",
        "2022-07",
        "2022-08",
        "2022-09",
        "2022-10",
        "2022-11",
        "2022-12",
        "2023-01",
    ]
    panel_rows: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []
    for community_type in ("general", "health"):
        for index in range(3):
            subreddit = f"{community_type}_{index}"
            for month_index, month in enumerate(months):
                panel_rows.append({
                    "subreddit": subreddit,
                    "month": month,
                    "community_type": community_type,
                    "comments": 100 + month_index,
                    "submissions": 10,
                    "comments_per_submission": 10.0 + month_index / 10,
                })
                interaction_rows.append({
                    "subreddit": subreddit,
                    "month": month,
                    "community_type": community_type,
                    "author_history_observed": int(month_index >= 2),
                    "new_author_share": 0.5,
                    "returning_author_share": 0.2,
                    "repeat_author_share": 0.3,
                    "op_return_rate": 0.4,
                    "reciprocal_dyad_share": 0.2,
                    "repeat_dyad_share": 0.3,
                    "multi_actor_thread_share": 0.5,
                    "focused_thread_share": 0.2,
                    "bond_index": float(month_index - 4),
                    "identity_index": float(4 - month_index),
                })
    panel_path = tmp_path / "panel.csv"
    interactions_path = tmp_path / "interactions.csv"
    pd.DataFrame(panel_rows).to_csv(panel_path, index=False)
    pd.DataFrame(interaction_rows).to_csv(interactions_path, index=False)

    result = run_interaction_outcomes_analysis(
        interactions_path=interactions_path,
        panel_path=panel_path,
        tables_dir=tmp_path / "tables",
    )

    assert set(result.summary["outcome"]) == {
        "bond_index",
        "identity_index",
        "new_author_share",
        "returning_author_share",
        "repeat_author_share",
        "op_return_rate",
        "reciprocal_dyad_share",
        "repeat_dyad_share",
        "multi_actor_thread_share",
        "focused_thread_share",
    }
    assert result.pretrend_tests["n_leads"].eq(3).all()
    assert result.table_paths["event_studies"].exists()
    assert result.summary["p_value_fdr"].between(0.0, 1.0).all()
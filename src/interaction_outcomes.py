"""Estimate descriptive DiD and event-study results for interaction outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR
from src.did import benjamini_hochberg, ModelSpec, estimate_twfe_did, run_event_study, run_pretrend_test
from src.panel import ensure_monthly_panel

INTERACTION_OUTCOMES = (
    "new_author_share",
    "returning_author_share",
    "repeat_author_share",
    "op_return_rate",
    "reciprocal_dyad_share",
    "repeat_dyad_share",
    "multi_actor_thread_share",
    "focused_thread_share",
    "bond_index",
    "identity_index",
)


@dataclass
class InteractionOutcomeArtifacts:
    """Tables written by the interaction-outcome analysis."""

    summary: pd.DataFrame
    pretrend_tests: pd.DataFrame
    event_studies: pd.DataFrame
    table_paths: dict[str, Path]


def run_interaction_outcomes_analysis(
    interactions_path: Path | None = None,
    panel_path: Path | None = None,
    *,
    tables_dir: Path | None = None,
) -> InteractionOutcomeArtifacts:
    """Estimate interaction outcomes after excluding author-history warm-up rows."""
    out_tables = tables_dir or TABLES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    if interactions_path is None:
        interactions_path = out_tables / "interactions-monthly.csv"
    if panel_path is None:
        panel_path, _, _ = ensure_monthly_panel(tables_dir=out_tables)

    interactions = pd.read_csv(interactions_path)
    panel = pd.read_csv(panel_path)
    frame = interactions.merge(
        panel[["subreddit", "month", "comments", "submissions", "comments_per_submission"]],
        on=["subreddit", "month"],
        how="inner",
    )
    frame = frame.loc[frame["author_history_observed"] == 1].copy()

    summary_rows: list[dict[str, object]] = []
    pretrend_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    for outcome in INTERACTION_OUTCOMES:
        for spec in (
            ModelSpec(model="unweighted"),
            ModelSpec(model="community_specific_trends", community_trends=True),
        ):
            summary_rows.append(
                estimate_twfe_did(frame, outcome, spec=spec, log_transform=False),
            )
        pretrend_rows.append(run_pretrend_test(frame, outcome, log_transform=False))
        event_frames.append(run_event_study(frame, outcome, log_transform=False))

    summary = pd.DataFrame(summary_rows).sort_values(["outcome", "model"], kind="stable")
    summary["p_value_fdr"] = summary.groupby("model", group_keys=False)["p_value"].apply(
        benjamini_hochberg,
    )
    pretrend_tests = pd.DataFrame(pretrend_rows).sort_values("outcome", kind="stable")
    event_studies = pd.concat(event_frames, ignore_index=True)
    table_paths = {
        "summary": out_tables / "interaction-outcomes-summary.csv",
        "pretrend_tests": out_tables / "interaction-outcomes-pretrend-tests.csv",
        "event_studies": out_tables / "interaction-outcomes-event-studies.csv",
    }
    summary.to_csv(table_paths["summary"], index=False)
    pretrend_tests.to_csv(table_paths["pretrend_tests"], index=False)
    event_studies.to_csv(table_paths["event_studies"], index=False)
    return InteractionOutcomeArtifacts(
        summary=summary,
        pretrend_tests=pretrend_tests,
        event_studies=event_studies,
        table_paths=table_paths,
    )
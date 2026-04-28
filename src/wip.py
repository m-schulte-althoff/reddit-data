"""Orchestrate the first-pass WIP analysis suite and summary outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_mentions import run_ai_mentions_analysis
from src.config import FIGURES_DIR, TABLES_DIR
from src.content_metrics import run_content_metrics_analysis
from src.did import run_did_analysis
from src.interactions import run_interactions_analysis
from src.mechanisms import run_mechanisms_analysis
from src.panel import ensure_monthly_panel
from src.responsiveness import run_responsiveness_analysis

WIP_KEY_RESULTS_CSV = "wip-key-results.csv"
WIP_KEY_RESULTS_MD = "wip-key-results.md"


@dataclass
class WipArtifacts:
    """Key result files plus the assembled summary table."""

    summary_table: pd.DataFrame
    csv_path: Path
    markdown_path: Path


def run_wip_suite(
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> WipArtifacts:
    """Run the main WIP suite and produce concise key-result outputs."""
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    panel_path, metadata_path, panel_metadata = ensure_monthly_panel(tables_dir=out_tables)
    panel = pd.read_csv(panel_path)
    did = run_did_analysis(panel_path=panel_path, tables_dir=out_tables, figures_dir=out_figures)
    responsiveness = run_responsiveness_analysis(tables_dir=out_tables, figures_dir=out_figures)
    mechanisms = run_mechanisms_analysis(panel_path=panel_path, tables_dir=out_tables, figures_dir=out_figures)
    ai_mentions = run_ai_mentions_analysis(tables_dir=out_tables, figures_dir=out_figures)
    content = run_content_metrics_analysis(tables_dir=out_tables, figures_dir=out_figures)
    interactions = run_interactions_analysis(tables_dir=out_tables, figures_dir=out_figures)

    summary_table = _build_key_results_table(
        panel,
        did.summary,
        responsiveness.monthly,
        ai_mentions.monthly,
        interactions.monthly,
    )
    csv_path = out_tables / WIP_KEY_RESULTS_CSV
    markdown_path = out_tables / WIP_KEY_RESULTS_MD
    summary_table.to_csv(csv_path, index=False)
    markdown_path.write_text(
        _build_markdown_summary(
            panel,
            panel_metadata,
            did.summary,
            responsiveness.monthly,
            mechanisms.summary,
            ai_mentions.monthly,
            content.monthly,
            interactions.monthly,
        ),
        encoding="utf-8",
    )

    return WipArtifacts(summary_table=summary_table, csv_path=csv_path, markdown_path=markdown_path)


def _build_key_results_table(
    panel: pd.DataFrame,
    did_summary: pd.DataFrame,
    responsiveness_monthly: pd.DataFrame,
    ai_monthly: pd.DataFrame,
    interactions_monthly: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append({
        "section": "panel",
        "metric": "n_subreddits",
        "group": "all",
        "value": float(panel["subreddit"].nunique()),
    })
    rows.append({
        "section": "panel",
        "metric": "n_months",
        "group": "all",
        "value": float(panel["month"].nunique()),
    })

    totals = (
        panel.groupby("community_type", as_index=False)[["comments", "submissions"]]
        .sum()
    )
    for _, row in totals.iterrows():
        rows.append({
            "section": "panel",
            "metric": "total_comments",
            "group": row["community_type"],
            "value": float(row["comments"]),
        })
        rows.append({
            "section": "panel",
            "metric": "total_submissions",
            "group": row["community_type"],
            "value": float(row["submissions"]),
        })

    did_main = did_summary.loc[did_summary["model"] == "unweighted"]
    for _, row in did_main.iterrows():
        rows.append({
            "section": "did",
            "metric": f"{row['outcome']}_estimate",
            "group": row["model"],
            "value": float(row["estimate"]),
        })

    resp_type = (
        responsiveness_monthly.loc[responsiveness_monthly["community_type"].isin(["health", "general"])]
        .groupby("community_type", as_index=False)[["reply_rate", "unanswered_rate", "op_followup_rate"]]
        .mean()
    )
    for _, row in resp_type.iterrows():
        rows.append({
            "section": "responsiveness",
            "metric": "reply_rate",
            "group": row["community_type"],
            "value": float(row["reply_rate"]),
        })
        rows.append({
            "section": "responsiveness",
            "metric": "unanswered_rate",
            "group": row["community_type"],
            "value": float(row["unanswered_rate"]),
        })

    ai_type = (
        ai_monthly.loc[ai_monthly["community_type"].isin(["health", "general"])]
        .groupby("community_type", as_index=False)[
            ["ai_mention_comment_share", "ai_mention_submission_share"]
        ]
        .mean()
    )
    for _, row in ai_type.iterrows():
        rows.append({
            "section": "ai_mentions",
            "metric": "comment_share",
            "group": row["community_type"],
            "value": float(row["ai_mention_comment_share"]),
        })
        rows.append({
            "section": "ai_mentions",
            "metric": "submission_share",
            "group": row["community_type"],
            "value": float(row["ai_mention_submission_share"]),
        })

    interaction_type = (
        interactions_monthly.loc[interactions_monthly["community_type"].isin(["health", "general"])]
        .groupby("community_type", as_index=False)[["bond_index", "identity_index"]]
        .mean()
    )
    for _, row in interaction_type.iterrows():
        rows.append({
            "section": "interactions",
            "metric": "bond_index",
            "group": row["community_type"],
            "value": float(row["bond_index"]),
        })
        rows.append({
            "section": "interactions",
            "metric": "identity_index",
            "group": row["community_type"],
            "value": float(row["identity_index"]),
        })

    return pd.DataFrame(rows)


def _build_markdown_summary(
    panel: pd.DataFrame,
    panel_metadata: dict[str, Any],
    did_summary: pd.DataFrame,
    responsiveness_monthly: pd.DataFrame,
    mechanisms_summary: pd.DataFrame,
    ai_monthly: pd.DataFrame,
    content_monthly: pd.DataFrame,
    interactions_monthly: pd.DataFrame,
) -> str:
    lines: list[str] = ["# WIP Key Results", ""]

    lines.append("## Coverage")
    lines.append(f"- Subreddits: {panel['subreddit'].nunique()}")
    lines.append(f"- Months covered: {panel['month'].min()} to {panel['month'].max()}")
    lines.append(f"- Panel metadata: {panel_metadata.get('n_rows', 0)} subreddit-month rows")
    lines.append("")

    lines.append("## Activity by Community Type")
    totals = panel.groupby("community_type", as_index=False)[["comments", "submissions"]].sum()
    for _, row in totals.iterrows():
        lines.append(
            f"- {row['community_type']}: {int(row['comments']):,} comments, {int(row['submissions']):,} submissions"
        )
    lines.append("")

    lines.append("## Main DiD Estimates")
    for outcome in ["comments", "submissions", "comments_per_submission"]:
        row = did_summary.loc[
            (did_summary["outcome"] == outcome) & (did_summary["model"] == "unweighted")
        ].iloc[0]
        lines.append(
            f"- {outcome}: estimate={row['estimate']:.4f}, SE={row['std_error']:.4f}, p={row['p_value']:.4f}"
        )
    lines.append("")

    lines.append("## Robustness Checks")
    for model in [
        "weighted_pre_activity",
        "balanced_panel",
        "exclude_top2_general",
        "winsorized_1_99",
        "cutoff_2023_03",
    ]:
        row = did_summary.loc[
            (did_summary["outcome"] == "comments") & (did_summary["model"] == model)
        ].iloc[0]
        lines.append(f"- comments / {model}: estimate={row['estimate']:.4f}, p={row['p_value']:.4f}")
    lines.append("")

    lines.append("## Responsiveness")
    resp_type = responsiveness_monthly.loc[
        responsiveness_monthly["community_type"].isin(["health", "general"])
    ].groupby("community_type", as_index=False)[["reply_rate", "unanswered_rate", "op_followup_rate"]].mean()
    for _, row in resp_type.iterrows():
        lines.append(
            f"- {row['community_type']}: reply_rate={row['reply_rate']:.3f}, unanswered_rate={row['unanswered_rate']:.3f}, op_followup_rate={row['op_followup_rate']:.3f}"
        )
    lines.append("")

    lines.append("## Helper Concentration")
    helper_type = panel.loc[panel["community_type"].isin(["health", "general"])]
    helper_type = helper_type.groupby("community_type", as_index=False)[["top5_share", "gini"]].mean()
    for _, row in helper_type.iterrows():
        lines.append(
            f"- {row['community_type']}: top5_share={row['top5_share']:.3f}, gini={row['gini']:.3f}"
        )
    lines.append("")

    lines.append("## AI Mentions")
    ai_type = ai_monthly.loc[ai_monthly["community_type"].isin(["health", "general"])]
    ai_type = ai_type.groupby("community_type", as_index=False)[["ai_mention_comment_share", "ai_mention_submission_share"]].mean()
    for _, row in ai_type.iterrows():
        lines.append(
            f"- {row['community_type']}: comment_share={row['ai_mention_comment_share']:.4f}, submission_share={row['ai_mention_submission_share']:.4f}"
        )
    lines.append("")

    lines.append("## Mechanisms")
    top_mech = mechanisms_summary.loc[
        mechanisms_summary["model_type"] == "triple_interaction"
    ].sort_values("p_value", kind="stable").head(5)
    for _, row in top_mech.iterrows():
        lines.append(
            f"- {row['outcome']} / {row['moderator']}: estimate={row['estimate']:.4f}, p={row['p_value']:.4f}"
        )
    lines.append("")

    lines.append("## Content Proxies")
    content_type = content_monthly.loc[content_monthly["community_type"].isin(["health", "general"])]
    content_type = content_type.groupby("community_type", as_index=False)[
        ["comment_question_share", "submission_experience_share", "comment_support_share"]
    ].mean()
    for _, row in content_type.iterrows():
        lines.append(
            f"- {row['community_type']}: comment_question_share={row['comment_question_share']:.3f}, submission_experience_share={row['submission_experience_share']:.3f}, comment_support_share={row['comment_support_share']:.3f}"
        )
    lines.append("")

    lines.append("## Interactions")
    interaction_type = interactions_monthly.loc[
        interactions_monthly["community_type"].isin(["health", "general"])
    ].groupby("community_type", as_index=False)[["bond_index", "identity_index"]].mean()
    for _, row in interaction_type.iterrows():
        lines.append(
            f"- {row['community_type']}: bond_index={row['bond_index']:.3f}, identity_index={row['identity_index']:.3f}"
        )
    lines.append("")

    return "\n".join(lines)
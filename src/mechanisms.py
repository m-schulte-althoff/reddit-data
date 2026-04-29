"""Moderator analyses for post-GenAI resilience mechanisms."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from src.config import FIGURES_DIR, TABLES_DIR
from src.did import BASELINE_CUTOFF
from src.panel import ensure_monthly_panel
from src.responsiveness import run_responsiveness_analysis
from src.thread_prep import ThreadPrepConfig

log = logging.getLogger(__name__)

MODERATOR_SOURCES: dict[str, str] = {
    "pre_top5_share": "top5_share",
    "pre_top1_share": "top1_share",
    "pre_gini": "gini",
    "pre_hhi": "hhi",
    "pre_pct1_share": "pct1_share",
    "pre_threading_ratio": "threading_ratio",
    "pre_mean_depth": "mean_depth",
    "pre_reply_rate": "reply_rate",
    "pre_unanswered_rate": "unanswered_rate",
    "pre_op_followup_rate": "op_followup_rate",
    "pre_median_latency": "median_first_reply_latency_hours",
}


@dataclass
class MechanismsArtifacts:
    """Written outputs plus in-memory moderation summary."""

    summary: pd.DataFrame
    merged_panel: pd.DataFrame
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]


def build_moderation_panel(
    panel: pd.DataFrame,
    responsiveness_monthly: pd.DataFrame,
    *,
    cutoff_month: str = BASELINE_CUTOFF,
) -> pd.DataFrame:
    """Merge panel and responsiveness metrics and add pre-period moderators."""
    merged = panel.merge(
        responsiveness_monthly,
        on=["subreddit", "month", "community_type"],
        how="left",
        suffixes=("", "_responsiveness"),
    )
    merged = merged.fillna(0.0)

    pre_period = merged.loc[merged["month"] < cutoff_month]
    moderator_means = (
        pre_period.groupby("subreddit", as_index=False)
        .agg({source: "mean" for source in MODERATOR_SOURCES.values()})
        .rename(columns={value: key for key, value in MODERATOR_SOURCES.items()})
    )

    return merged.merge(moderator_means, on="subreddit", how="left").fillna(0.0)


def estimate_moderation_model(
    merged_panel: pd.DataFrame,
    outcome_column: str,
    moderator_column: str,
    *,
    health_only: bool = False,
    cutoff_month: str = BASELINE_CUTOFF,
) -> dict[str, Any]:
    """Estimate one moderation specification with subreddit-clustered SEs."""
    frame = merged_panel.loc[
        merged_panel["community_type"].isin(["general", "health"]),
    ].copy()
    frame["health"] = (frame["community_type"] == "health").astype(int)
    frame["post"] = (frame["month"] >= cutoff_month).astype(int)
    frame["moderator"] = frame[moderator_column].astype(float)
    frame["outcome"] = frame[outcome_column].astype(float).map(lambda value: math.log1p(max(value, 0.0)))

    if health_only:
        frame = frame.loc[frame["health"] == 1].copy()
        frame["post_moderator"] = frame["post"] * frame["moderator"]
        formula = "outcome ~ post_moderator + C(subreddit) + C(month)"
        parameter = "post_moderator"
        model_type = "within_health"
    else:
        frame["health_post"] = frame["health"] * frame["post"]
        frame["post_moderator"] = frame["post"] * frame["moderator"]
        frame["health_post_moderator"] = frame["health"] * frame["post"] * frame["moderator"]
        formula = "outcome ~ health_post + post_moderator + health_post_moderator + C(subreddit) + C(month)"
        parameter = "health_post_moderator"
        model_type = "triple_interaction"

    fitted = smf.ols(formula, data=frame).fit(
        cov_type="cluster",
        cov_kwds={"groups": frame["subreddit"]},
    )

    return {
        "outcome": outcome_column,
        "moderator": moderator_column,
        "model_type": model_type,
        "estimate": float(fitted.params[parameter]),
        "std_error": float(fitted.bse[parameter]),
        "p_value": float(fitted.pvalues[parameter]),
        "n_obs": int(len(frame)),
        "n_subreddits": int(frame["subreddit"].nunique()),
    }


def run_mechanisms_analysis(
    panel_path: Path | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> MechanismsArtifacts:
    """Run moderator regressions and write summary figures."""
    if panel_path is None:
        if thread_prep is None:
            panel_path, _, _ = ensure_monthly_panel()
        else:
            panel_path, _, _ = ensure_monthly_panel(thread_prep=thread_prep)
    panel = pd.read_csv(panel_path)
    if thread_prep is None:
        responsiveness = run_responsiveness_analysis()
    else:
        responsiveness = run_responsiveness_analysis(thread_prep=thread_prep)
    merged_panel = build_moderation_panel(panel, responsiveness.monthly)

    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for outcome in ["comments", "submissions", "comments_per_submission"]:
        for moderator in MODERATOR_SOURCES:
            summary_rows.append(estimate_moderation_model(merged_panel, outcome, moderator))
            summary_rows.append(
                estimate_moderation_model(
                    merged_panel,
                    outcome,
                    moderator,
                    health_only=True,
                ),
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["model_type", "outcome", "moderator"],
        kind="stable",
    )

    table_paths = {
        "summary": out_tables / "mechanism-moderation-summary.csv",
    }
    summary.to_csv(table_paths["summary"], index=False)

    figure_paths = {
        "coefficients": out_figures / "mechanism-moderation-coefficients.svg",
        "top5": out_figures / "mechanism-high-low-trends-top5.svg",
        "reply_rate": out_figures / "mechanism-high-low-trends-reply-rate.svg",
        "threading": out_figures / "mechanism-high-low-trends-threading.svg",
    }
    _plot_moderation_coefficients(summary, figure_paths["coefficients"])
    _plot_high_low_trend(merged_panel, "pre_top5_share", "Top-5 helper share", figure_paths["top5"])
    _plot_high_low_trend(merged_panel, "pre_reply_rate", "Reply rate", figure_paths["reply_rate"])
    _plot_high_low_trend(
        merged_panel,
        "pre_threading_ratio",
        "Threading ratio",
        figure_paths["threading"],
    )

    log.info("Wrote %s", table_paths["summary"])
    return MechanismsArtifacts(
        summary=summary.reset_index(drop=True),
        merged_panel=merged_panel,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )


def _plot_moderation_coefficients(summary: pd.DataFrame, out_path: Path) -> None:
    plot_frame = summary.loc[summary["model_type"] == "triple_interaction"].copy()
    outcomes = ["comments", "submissions", "comments_per_submission"]
    colors = {
        "comments": "#1b9e77",
        "submissions": "#d95f02",
        "comments_per_submission": "#7570b3",
    }

    moderators = list(MODERATOR_SOURCES)
    x_positions = list(range(len(moderators)))
    offsets = {
        "comments": -0.2,
        "submissions": 0.0,
        "comments_per_submission": 0.2,
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    for outcome in outcomes:
        subset = plot_frame.loc[plot_frame["outcome"] == outcome].set_index("moderator")
        xs = [position + offsets[outcome] for position in x_positions]
        ys = [subset.loc[moderator, "estimate"] for moderator in moderators]
        errs = [1.96 * subset.loc[moderator, "std_error"] for moderator in moderators]
        ax.errorbar(xs, ys, yerr=errs, fmt="o", color=colors[outcome], label=outcome, capsize=3)

    ax.set_xticks(x_positions, [moderator.replace("pre_", "") for moderator in moderators], rotation=45)
    ax.set_ylabel("Triple-interaction coefficient")
    ax.set_title("Mechanism moderation estimates")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_high_low_trend(
    merged_panel: pd.DataFrame,
    moderator_column: str,
    label: str,
    out_path: Path,
) -> None:
    moderator_values = (
        merged_panel[["subreddit", moderator_column]]
        .drop_duplicates(subset=["subreddit"])
        .set_index("subreddit")[moderator_column]
    )
    cutoff = float(moderator_values.median())
    groups = moderator_values.map(lambda value: "High" if value >= cutoff else "Low")

    plot_frame = merged_panel.copy()
    plot_frame["moderator_group"] = plot_frame["subreddit"].map(groups)
    trend = (
        plot_frame.groupby(["month", "moderator_group"], as_index=False)["comments"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    styles = {
        "High": {"color": "#1b9e77", "marker": "o"},
        "Low": {"color": "#d95f02", "marker": "s"},
    }
    for group in ["High", "Low"]:
        subset = trend.loc[trend["moderator_group"] == group]
        ax.plot(
            subset["month"],
            subset["comments"],
            color=styles[group]["color"],
            marker=styles[group]["marker"],
            linewidth=2,
            label=group,
        )

    ax.axvline(BASELINE_CUTOFF, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean comments per subreddit")
    ax.set_title(f"High vs. low pre-period {label}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
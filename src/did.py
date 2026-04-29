"""Difference-in-differences and event-study analysis on the monthly panel."""

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
from src.io_utils import months_since
from src.panel import ensure_monthly_panel
from src.thread_prep import ThreadPrepConfig

log = logging.getLogger(__name__)

BASELINE_CUTOFF = "2022-11"
EVENT_STUDY_LOWER_BIN = -6
EVENT_STUDY_UPPER_BIN = 24


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one DiD specification."""

    model: str
    cutoff_month: str = BASELINE_CUTOFF
    weighted: bool = False
    balanced_only: bool = False
    exclude_top2_general: bool = False
    winsorize: bool = False


@dataclass
class DidAnalysisArtifacts:
    """Written artifacts plus in-memory tables for testing and reuse."""

    summary: pd.DataFrame
    event_studies: dict[str, pd.DataFrame]
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]


def load_panel_dataframe(
    panel_path: Path | None = None,
    *,
    thread_prep: ThreadPrepConfig | None = None,
) -> pd.DataFrame:
    """Load the monthly panel, building it first when necessary."""
    if panel_path is None:
        panel_path, _, _ = ensure_monthly_panel(thread_prep=thread_prep)

    frame = pd.read_csv(panel_path)
    frame = frame.sort_values(["subreddit", "month"], kind="stable").reset_index(drop=True)
    return frame


def prepare_analysis_frame(
    panel: pd.DataFrame,
    outcome_column: str,
    *,
    cutoff_month: str = BASELINE_CUTOFF,
    balanced_only: bool = False,
    exclude_top2_general: bool = False,
    winsorize: bool = False,
) -> pd.DataFrame:
    """Prepare a regression-ready frame for one outcome and specification."""
    frame = panel.loc[
        panel["community_type"].isin(["general", "health"]),
    ].copy()
    frame = frame.sort_values(["subreddit", "month"], kind="stable")

    if balanced_only:
        month_count = frame["month"].nunique()
        balanced_subreddits = (
            frame.groupby("subreddit")["month"]
            .nunique()
            .loc[lambda s: s == month_count]
            .index
        )
        frame = frame.loc[frame["subreddit"].isin(balanced_subreddits)].copy()

    if exclude_top2_general:
        top_general = (
            frame.loc[
                (frame["community_type"] == "general") & (frame["month"] < cutoff_month),
                ["subreddit", "comments"],
            ]
            .groupby("subreddit", as_index=True)["comments"]
            .sum()
            .nlargest(2)
            .index
        )
        frame = frame.loc[~frame["subreddit"].isin(top_general)].copy()

    frame["health"] = (frame["community_type"] == "health").astype(int)
    frame["post"] = (frame["month"] >= cutoff_month).astype(int)

    raw_outcome = frame[outcome_column].astype(float).fillna(0.0)
    if winsorize and not raw_outcome.empty:
        lower = raw_outcome.quantile(0.01)
        upper = raw_outcome.quantile(0.99)
        raw_outcome = raw_outcome.clip(lower=lower, upper=upper)

    frame["raw_outcome"] = raw_outcome
    frame["outcome"] = raw_outcome.map(lambda value: math.log1p(max(value, 0.0)))
    frame["health_post"] = frame["health"] * frame["post"]

    pre_activity = (
        frame.loc[frame["month"] < cutoff_month]
        .groupby("subreddit")[["comments", "submissions"]]
        .sum()
        .sum(axis=1)
    )
    weights = frame["subreddit"].map(pre_activity).fillna(0.0).clip(lower=1.0)
    frame["pre_activity_weight"] = weights
    return frame.reset_index(drop=True)


def estimate_twfe_did(
    panel: pd.DataFrame,
    outcome_column: str,
    *,
    spec: ModelSpec,
) -> dict[str, Any]:
    """Estimate a two-way fixed-effects DiD model with clustered SEs."""
    frame = prepare_analysis_frame(
        panel,
        outcome_column,
        cutoff_month=spec.cutoff_month,
        balanced_only=spec.balanced_only,
        exclude_top2_general=spec.exclude_top2_general,
        winsorize=spec.winsorize,
    )
    if frame.empty:
        raise ValueError(f"No data available for {outcome_column} / {spec.model}")

    formula = "outcome ~ health_post + C(subreddit) + C(month)"
    if spec.weighted:
        fitted = smf.wls(
            formula,
            data=frame,
            weights=frame["pre_activity_weight"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame["subreddit"]})
    else:
        fitted = smf.ols(formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["subreddit"]},
        )

    return {
        "outcome": outcome_column,
        "model": spec.model,
        "estimate": float(fitted.params["health_post"]),
        "std_error": float(fitted.bse["health_post"]),
        "p_value": float(fitted.pvalues["health_post"]),
        "n_obs": int(len(frame)),
        "n_subreddits": int(frame["subreddit"].nunique()),
        "mean_pre_health": _safe_mean(
            frame.loc[(frame["health"] == 1) & (frame["post"] == 0), "raw_outcome"],
        ),
        "mean_post_health": _safe_mean(
            frame.loc[(frame["health"] == 1) & (frame["post"] == 1), "raw_outcome"],
        ),
        "mean_pre_general": _safe_mean(
            frame.loc[(frame["health"] == 0) & (frame["post"] == 0), "raw_outcome"],
        ),
        "mean_post_general": _safe_mean(
            frame.loc[(frame["health"] == 0) & (frame["post"] == 1), "raw_outcome"],
        ),
        "weighted_change_health": _weighted_change(frame, health=1),
        "weighted_change_general": _weighted_change(frame, health=0),
        "cutoff_month": spec.cutoff_month,
        "weighted": int(spec.weighted),
        "balanced_only": int(spec.balanced_only),
        "exclude_top2_general": int(spec.exclude_top2_general),
        "winsorized": int(spec.winsorize),
    }


def run_event_study(
    panel: pd.DataFrame,
    outcome_column: str,
    *,
    cutoff_month: str = BASELINE_CUTOFF,
    weighted: bool = False,
) -> pd.DataFrame:
    """Estimate an event-study specification for one outcome."""
    frame = prepare_analysis_frame(panel, outcome_column, cutoff_month=cutoff_month)
    frame["event_time"] = frame["month"].map(lambda month: months_since(month, reference=cutoff_month))
    frame["event_bin"] = frame["event_time"].map(_bin_event_time)

    terms: list[str] = []
    column_names: dict[int, str] = {}
    for event_time in sorted(frame["event_bin"].unique()):
        if int(event_time) == -1:
            continue
        column_name = f"health_event_{_event_token(int(event_time))}"
        frame[column_name] = ((frame["event_bin"] == event_time) & (frame["health"] == 1)).astype(int)
        terms.append(column_name)
        column_names[int(event_time)] = column_name

    formula = "outcome ~ " + " + ".join(terms) + " + C(subreddit) + C(month)"
    if weighted:
        fitted = smf.wls(
            formula,
            data=frame,
            weights=frame["pre_activity_weight"],
        ).fit(cov_type="cluster", cov_kwds={"groups": frame["subreddit"]})
    else:
        fitted = smf.ols(formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["subreddit"]},
        )

    rows: list[dict[str, Any]] = []
    for event_time in sorted(column_names):
        column_name = column_names[event_time]
        estimate = float(fitted.params[column_name])
        std_error = float(fitted.bse[column_name])
        rows.append({
            "outcome": outcome_column,
            "event_time": event_time,
            "event_label": _event_label(event_time),
            "estimate": estimate,
            "std_error": std_error,
            "p_value": float(fitted.pvalues[column_name]),
            "ci_low": estimate - 1.96 * std_error,
            "ci_high": estimate + 1.96 * std_error,
            "n_obs": int(len(frame)),
            "n_subreddits": int(frame["subreddit"].nunique()),
            "cutoff_month": cutoff_month,
        })

    return pd.DataFrame(rows).sort_values(["event_time"], kind="stable").reset_index(drop=True)


def run_did_analysis(
    panel_path: Path | None = None,
    *,
    tables_dir: Path | None = None,
    figures_dir: Path | None = None,
    thread_prep: ThreadPrepConfig | None = None,
) -> DidAnalysisArtifacts:
    """Run the full DiD and event-study suite and write tables/figures."""
    panel = load_panel_dataframe(panel_path, thread_prep=thread_prep)
    out_tables = tables_dir or TABLES_DIR
    out_figures = figures_dir or FIGURES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    specs = [
        ModelSpec(model="unweighted"),
        ModelSpec(model="weighted_pre_activity", weighted=True),
        ModelSpec(model="balanced_panel", balanced_only=True),
        ModelSpec(model="exclude_top2_general", exclude_top2_general=True),
        ModelSpec(model="winsorized_1_99", winsorize=True),
        ModelSpec(model="cutoff_2023_03", cutoff_month="2023-03"),
        ModelSpec(model="cutoff_2023_07", cutoff_month="2023-07"),
    ]
    outcomes = ["comments", "submissions", "comments_per_submission"]

    summary_rows: list[dict[str, Any]] = []
    for outcome_column in outcomes:
        for spec in specs:
            summary_rows.append(estimate_twfe_did(panel, outcome_column, spec=spec))

    summary = pd.DataFrame(summary_rows).sort_values(["outcome", "model"], kind="stable")

    event_studies = {
        "comments": run_event_study(panel, "comments"),
        "submissions": run_event_study(panel, "submissions"),
        "comments_per_submission": run_event_study(panel, "comments_per_submission"),
    }

    table_paths = {
        "summary": out_tables / "did-summary.csv",
        "event_comments": out_tables / "did-event-study-comments.csv",
        "event_submissions": out_tables / "did-event-study-submissions.csv",
        "event_comments_per_submission": (
            out_tables / "did-event-study-comments-per-submission.csv"
        ),
    }
    summary.to_csv(table_paths["summary"], index=False)
    event_studies["comments"].to_csv(table_paths["event_comments"], index=False)
    event_studies["submissions"].to_csv(table_paths["event_submissions"], index=False)
    event_studies["comments_per_submission"].to_csv(
        table_paths["event_comments_per_submission"],
        index=False,
    )

    figure_paths = {
        "trend_comments": out_figures / "did-trends-comments-health-vs-general.svg",
        "trend_submissions": out_figures / "did-trends-submissions-health-vs-general.svg",
        "event_comments": out_figures / "did-event-study-comments.svg",
        "event_submissions": out_figures / "did-event-study-submissions.svg",
        "event_comments_per_submission": (
            out_figures / "did-event-study-comments-per-submission.svg"
        ),
    }

    _plot_health_general_trends(panel, "comments", figure_paths["trend_comments"])
    _plot_health_general_trends(panel, "submissions", figure_paths["trend_submissions"])
    _plot_event_study(event_studies["comments"], figure_paths["event_comments"], "Comments")
    _plot_event_study(
        event_studies["submissions"],
        figure_paths["event_submissions"],
        "Submissions",
    )
    _plot_event_study(
        event_studies["comments_per_submission"],
        figure_paths["event_comments_per_submission"],
        "Comments per submission",
    )

    log.info("Wrote %s", table_paths["summary"])
    return DidAnalysisArtifacts(
        summary=summary.reset_index(drop=True),
        event_studies=event_studies,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = float(series.mean())
    return 0.0 if pd.isna(value) else value


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    if values.empty:
        return 0.0
    total_weight = float(weights.sum())
    if total_weight == 0:
        return 0.0
    return float((values * weights).sum() / total_weight)


def _weighted_change(frame: pd.DataFrame, *, health: int) -> float:
    subset = frame.loc[frame["health"] == health]
    pre = subset.loc[subset["post"] == 0]
    post = subset.loc[subset["post"] == 1]
    return _weighted_mean(post["raw_outcome"], post["pre_activity_weight"]) - _weighted_mean(
        pre["raw_outcome"],
        pre["pre_activity_weight"],
    )


def _bin_event_time(event_time: int) -> int:
    if event_time <= EVENT_STUDY_LOWER_BIN:
        return EVENT_STUDY_LOWER_BIN
    if event_time >= EVENT_STUDY_UPPER_BIN:
        return EVENT_STUDY_UPPER_BIN
    return int(event_time)


def _event_token(event_time: int) -> str:
    if event_time <= EVENT_STUDY_LOWER_BIN:
        return "le_m6"
    if event_time >= EVENT_STUDY_UPPER_BIN:
        return "ge_p24"
    if event_time < 0:
        return f"m{abs(event_time)}"
    return f"p{event_time}"


def _event_label(event_time: int) -> str:
    if event_time <= EVENT_STUDY_LOWER_BIN:
        return "<=-6"
    if event_time >= EVENT_STUDY_UPPER_BIN:
        return ">=24"
    return str(event_time)


def _plot_health_general_trends(panel: pd.DataFrame, outcome: str, out_path: Path) -> None:
    plot_frame = (
        panel.loc[panel["community_type"].isin(["general", "health"])]
        .groupby(["month", "community_type"], as_index=False)[outcome]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    styles = {
        "health": {"label": "Health", "color": "#1b9e77", "marker": "o"},
        "general": {"label": "General", "color": "#d95f02", "marker": "s"},
    }

    for community_type in ["health", "general"]:
        subset = plot_frame.loc[plot_frame["community_type"] == community_type]
        if subset.empty:
            continue
        style = styles[community_type]
        ax.plot(
            subset["month"],
            subset[outcome],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
        )

    ax.axvline(BASELINE_CUTOFF, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Total {outcome.replace('_', ' ')}")
    ax.set_title(f"Monthly {outcome.replace('_', ' ')} by community type")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_event_study(event_frame: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.errorbar(
        event_frame["event_time"],
        event_frame["estimate"],
        yerr=1.96 * event_frame["std_error"],
        fmt="o-",
        color="#1f78b4",
        ecolor="#6baed6",
        capsize=3,
    )
    ax.set_xticks(event_frame["event_time"].tolist(), event_frame["event_label"].tolist(), rotation=45)
    ax.set_xlabel("Months relative to 2022-11")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Event study: {title}")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
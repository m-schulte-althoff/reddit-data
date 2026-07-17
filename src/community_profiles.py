"""Exploratory health-community growth profiles and pre-period structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR
from src.did import BASELINE_CUTOFF


@dataclass
class CommunityProfileArtifacts:
    """Profile and grouped-summary tables for health communities."""

    profiles: pd.DataFrame
    summary: pd.DataFrame
    table_paths: dict[str, Path]


def run_community_profile_analysis(
    panel_path: Path | None = None,
    interactions_path: Path | None = None,
    *,
    tables_dir: Path | None = None,
) -> CommunityProfileArtifacts:
    """Describe health-community activity changes and their prior structure."""
    out_tables = tables_dir or TABLES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    resolved_panel = panel_path or (out_tables / "community-monthly-panel.csv")
    resolved_interactions = interactions_path or (out_tables / "interactions-monthly.csv")
    panel = pd.read_csv(resolved_panel)
    health = panel.loc[panel["community_type"] == "health"].copy()
    health["post"] = health["month"] >= BASELINE_CUTOFF

    profile = health.groupby("subreddit", as_index=False).agg(
        pre_mean_comments=("comments", lambda values: values[health.loc[values.index, "post"].eq(False)].mean()),
        post_mean_comments=("comments", lambda values: values[health.loc[values.index, "post"].eq(True)].mean()),
        pre_mean_submissions=("submissions", lambda values: values[health.loc[values.index, "post"].eq(False)].mean()),
        post_mean_submissions=("submissions", lambda values: values[health.loc[values.index, "post"].eq(True)].mean()),
        pre_threading_ratio=("threading_ratio", lambda values: values[health.loc[values.index, "post"].eq(False)].mean()),
        pre_gini=("gini", lambda values: values[health.loc[values.index, "post"].eq(False)].mean()),
        pre_top5_share=("top5_share", lambda values: values[health.loc[values.index, "post"].eq(False)].mean()),
    )
    profile["comment_change_pct"] = (
        (profile["post_mean_comments"] - profile["pre_mean_comments"])
        / profile["pre_mean_comments"].clip(lower=1.0)
        * 100.0
    )
    profile["submission_change_pct"] = (
        (profile["post_mean_submissions"] - profile["pre_mean_submissions"])
        / profile["pre_mean_submissions"].clip(lower=1.0)
        * 100.0
    )
    profile["growth_profile"] = _growth_profiles(profile["comment_change_pct"])
    profile["analysis_label"] = "exploratory_descriptive"

    if resolved_interactions.exists():
        interactions = pd.read_csv(resolved_interactions)
        interactions = interactions.loc[
            (interactions["community_type"] == "health")
            & (interactions["month"] < BASELINE_CUTOFF)
            & (interactions["author_history_observed"] == 1),
        ]
        interaction_means = interactions.groupby("subreddit", as_index=False).agg(
            pre_bond_index=("bond_index", "mean"),
            pre_identity_index=("identity_index", "mean"),
            pre_reciprocal_dyad_share=("reciprocal_dyad_share", "mean"),
        )
        profile = profile.merge(interaction_means, on="subreddit", how="left")

    summary = profile.groupby("growth_profile", as_index=False).agg(
        n_subreddits=("subreddit", "count"),
        mean_comment_change_pct=("comment_change_pct", "mean"),
        mean_pre_threading_ratio=("pre_threading_ratio", "mean"),
        mean_pre_gini=("pre_gini", "mean"),
        mean_pre_top5_share=("pre_top5_share", "mean"),
    )
    table_paths = {
        "profiles": out_tables / "health-community-profiles.csv",
        "summary": out_tables / "health-community-profile-summary.csv",
    }
    profile.sort_values("subreddit", kind="stable").to_csv(table_paths["profiles"], index=False)
    summary.sort_values("growth_profile", kind="stable").to_csv(table_paths["summary"], index=False)
    return CommunityProfileArtifacts(profiles=profile, summary=summary, table_paths=table_paths)


def _growth_profiles(change: pd.Series) -> pd.Series:
    """Assign stable tercile labels without relying on quantile-edge uniqueness."""
    ranks = change.rank(method="first", pct=True)
    return pd.Series(
        pd.cut(
            ranks,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["declining_or_low_growth", "moderate_growth", "high_growth"],
            include_lowest=True,
        ).astype(str),
        index=change.index,
    )
"""Run the full filtered-data analysis stack and write a combined summary."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.ai_mentions import run_ai_mentions_analysis
from src.config import OUTPUT_DIR
from src.content_metrics import run_content_metrics_analysis
from src.describe import describe_filtered
from src.did import run_did_analysis
from src.discursivity import compute_discursivity, load_discursivity, save_discursivity
from src.helpers import _discover_filtered_paths, analyse_helpers, compute_helpers
from src.interactions import run_interactions_analysis
from src.mechanisms import run_mechanisms_analysis
from src.panel import ensure_monthly_panel
from src.resilience import compute_resilience
from src.responsiveness import run_responsiveness_analysis
from src.wip import run_wip_suite

SUMMARY_FILENAME = "summary.md"


@dataclass(frozen=True)
class AnalysisSection:
    """One logical analysis section for the combined summary."""

    title: str
    description: str
    table_paths: list[Path]
    figure_paths: list[Path]


@dataclass(frozen=True)
class AllAnalysisArtifacts:
    """Artifacts produced by the filtered-data analysis pipeline."""

    summary_path: Path
    sections: list[AnalysisSection]


def allAnalysis(*, output_dir: Path | None = None) -> AllAnalysisArtifacts:
    """Run all analyses that depend on already-filtered data.

    This pipeline assumes filtered data already exists in ``data/processed/`` and
    runs the descriptive, panel, causal, mechanism, and summary layers.
    """
    out_output = output_dir or OUTPUT_DIR
    out_tables = out_output / "tables"
    out_figures = out_output / "figures"
    out_output.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    sections: list[AnalysisSection] = []
    sections.append(run_describe_outputs(tables_dir=out_tables, figures_dir=out_figures))
    sections.append(run_discursivity_outputs(tables_dir=out_tables, figures_dir=out_figures))
    sections.append(run_helpers_outputs(tables_dir=out_tables, figures_dir=out_figures))
    sections.append(run_resilience_outputs(tables_dir=out_tables, figures_dir=out_figures))

    panel_csv_path, panel_metadata_path, _ = ensure_monthly_panel(tables_dir=out_tables)
    sections.append(
        AnalysisSection(
            title="Panel",
            description="Monthly subreddit panel for all downstream analyses.",
            table_paths=[panel_csv_path, panel_metadata_path],
            figure_paths=[],
        ),
    )

    did = run_did_analysis(panel_path=panel_csv_path, tables_dir=out_tables, figures_dir=out_figures)
    sections.append(
        AnalysisSection(
            title="DiD and Event Study",
            description="Fixed-effects DiD, robustness checks, and event-study outputs.",
            table_paths=list(did.table_paths.values()),
            figure_paths=list(did.figure_paths.values()),
        ),
    )

    responsiveness = run_responsiveness_analysis(tables_dir=out_tables, figures_dir=out_figures)
    sections.append(
        AnalysisSection(
            title="Responsiveness",
            description="Reply timing, support availability, and OP follow-up metrics.",
            table_paths=list(responsiveness.table_paths.values()),
            figure_paths=list(responsiveness.figure_paths.values()),
        ),
    )

    mechanisms = run_mechanisms_analysis(
        panel_path=panel_csv_path,
        tables_dir=out_tables,
        figures_dir=out_figures,
    )
    sections.append(
        AnalysisSection(
            title="Mechanisms",
            description="Moderator models linking pre-period structure to post-GenAI change.",
            table_paths=list(mechanisms.table_paths.values()),
            figure_paths=list(mechanisms.figure_paths.values()),
        ),
    )

    ai_mentions = run_ai_mentions_analysis(tables_dir=out_tables, figures_dir=out_figures)
    sections.append(
        AnalysisSection(
            title="AI Mentions",
            description="Regex-based monthly GenAI mention counts and trends.",
            table_paths=list(ai_mentions.table_paths.values()),
            figure_paths=list(ai_mentions.figure_paths.values()),
        ),
    )

    content = run_content_metrics_analysis(tables_dir=out_tables, figures_dir=out_figures)
    sections.append(
        AnalysisSection(
            title="Content Metrics",
            description="Lightweight effort, experience, support, and information proxies.",
            table_paths=list(content.table_paths.values()),
            figure_paths=list(content.figure_paths.values()),
        ),
    )

    interactions = run_interactions_analysis(tables_dir=out_tables, figures_dir=out_figures)
    sections.append(
        AnalysisSection(
            title="Interactions",
            description="Bond-vs-identity author, dyad, and thread-structure metrics.",
            table_paths=list(interactions.table_paths.values()),
            figure_paths=list(interactions.figure_paths.values()),
        ),
    )

    wip = run_wip_suite(tables_dir=out_tables, figures_dir=out_figures)
    sections.append(
        AnalysisSection(
            title="WIP Key Results",
            description="Condensed manuscript-oriented summary of the main post-filter results.",
            table_paths=[wip.csv_path, wip.markdown_path],
            figure_paths=[],
        ),
    )

    summary_path = out_output / SUMMARY_FILENAME
    summary_path.write_text(
        _build_summary_markdown(
            output_dir=out_output,
            sections=sections,
            wip_markdown_path=wip.markdown_path,
        ),
        encoding="utf-8",
    )
    return AllAnalysisArtifacts(summary_path=summary_path, sections=sections)


def run_all_analysis(*, output_dir: Path | None = None) -> AllAnalysisArtifacts:
    """Snake-case alias for ``allAnalysis``."""
    return allAnalysis(output_dir=output_dir)


def run_describe_outputs(*, tables_dir: Path, figures_dir: Path) -> AnalysisSection:
    """Run the descriptive overview for comments and submissions."""
    import views

    table_paths: list[Path] = []
    figure_paths: list[Path] = []
    with _override_view_dirs(tables_dir=tables_dir, figures_dir=figures_dir):
        for kind in ("comments", "submissions"):
            result = describe_filtered(kind)
            if result.total_records == 0:
                continue
            table_paths.extend([
                views.write_describe_summary_csv(result, f"describe-{kind}-summary.csv"),
                views.write_describe_monthly_csv(result, f"describe-{kind}-monthly.csv"),
                views.write_describe_monthly_csv(
                    result,
                    f"describe-{kind}-monthly-top15.csv",
                    top_n=15,
                ),
            ])
            figure_paths.extend([
                views.plot_describe_trend_aggregated(
                    result,
                    f"describe-{kind}-trend-aggregated.svg",
                ),
                views.plot_describe_trend_by_community_type(
                    result,
                    f"describe-{kind}-trend-community-types.svg",
                ),
                views.plot_describe_trend_per_subreddit(
                    result,
                    f"describe-{kind}-trend-all.svg",
                    top_n=None,
                ),
                views.plot_describe_trend_per_subreddit(
                    result,
                    f"describe-{kind}-trend-top15.svg",
                ),
            ])

    return AnalysisSection(
        title="Describe",
        description="Descriptive overviews of filtered comments and submissions.",
        table_paths=table_paths,
        figure_paths=figure_paths,
    )


def run_discursivity_outputs(*, tables_dir: Path, figures_dir: Path) -> AnalysisSection:
    """Run discursivity outputs with cache persistence."""
    import views

    result = compute_discursivity()
    if result.resolved_comments == 0:
        return AnalysisSection(
            title="Discursivity",
            description="No resolved comments were available for discursivity outputs.",
            table_paths=[],
            figure_paths=[],
        )

    save_discursivity(
        result,
        comment_paths=_discover_filtered_paths("comments"),
        submission_paths=_discover_filtered_paths("submissions"),
        out_dir=tables_dir,
    )
    with _override_view_dirs(tables_dir=tables_dir, figures_dir=figures_dir):
        table_paths = [views.write_discursivity_csv(result, "discursivity-monthly.csv")]
        figure_paths = [
            views.plot_discursivity_mean_depth(result, "discursivity-mean-depth-top15.svg"),
            views.plot_discursivity_mean_depth(
                result,
                "discursivity-mean-depth-all.svg",
                top_n=None,
            ),
            views.plot_discursivity_threading_ratio(
                result,
                "discursivity-threading-ratio-top15.svg",
            ),
            views.plot_discursivity_threading_ratio(
                result,
                "discursivity-threading-ratio-all.svg",
                top_n=None,
            ),
        ]

    return AnalysisSection(
        title="Discursivity",
        description="Comment-depth and threading-ratio outputs.",
        table_paths=table_paths + [tables_dir / "discursivity-cache.json"],
        figure_paths=figure_paths,
    )


def run_helpers_outputs(*, tables_dir: Path, figures_dir: Path) -> AnalysisSection:
    """Run repeat-helper concentration outputs."""
    import views

    result = compute_helpers(_discover_filtered_paths("comments"))
    if not result.cells:
        return AnalysisSection(
            title="Helpers",
            description="No helper concentration outputs were generated.",
            table_paths=[],
            figure_paths=[],
        )

    analysis = analyse_helpers(result)
    with _override_view_dirs(tables_dir=tables_dir, figures_dir=figures_dir):
        table_paths = [
            views.write_helpers_monthly_csv(result, "helpers-monthly.csv"),
            views.write_helpers_type_summary_csv(analysis, "helpers-type-summary.csv"),
            views.write_helpers_moderation_csv(analysis, "helpers-moderation.csv"),
        ]
        figure_paths = [
            views.plot_helpers_type_comparison(analysis, "helpers-type-comparison.svg"),
            views.plot_helpers_moderation_scatter(
                analysis,
                "helpers-moderation-gini.svg",
                metric="gini",
            ),
            views.plot_helpers_moderation_scatter(
                analysis,
                "helpers-moderation-top5.svg",
                metric="top5_share",
            ),
            views.plot_helpers_gini_trend(result, "helpers-gini-trend.svg"),
        ]

    return AnalysisSection(
        title="Helpers",
        description="Repeat-helper concentration and moderation outputs.",
        table_paths=table_paths,
        figure_paths=figure_paths,
    )


def run_resilience_outputs(*, tables_dir: Path, figures_dir: Path) -> AnalysisSection:
    """Run resilience outputs using the local discursivity cache when valid."""
    import views

    disc = load_discursivity(
        _discover_filtered_paths("comments"),
        _discover_filtered_paths("submissions"),
        cache_dir=tables_dir,
    )
    if disc is None:
        disc = compute_discursivity()
        if disc.resolved_comments > 0:
            save_discursivity(
                disc,
                comment_paths=_discover_filtered_paths("comments"),
                submission_paths=_discover_filtered_paths("submissions"),
                out_dir=tables_dir,
            )
    if disc.resolved_comments == 0:
        return AnalysisSection(
            title="Resilience",
            description="No resilience outputs were generated because no resolved comments were available.",
            table_paths=[],
            figure_paths=[],
        )

    result = compute_resilience(disc)
    if not result.profiles:
        return AnalysisSection(
            title="Resilience",
            description="No qualifying subreddits were available for resilience outputs.",
            table_paths=[],
            figure_paths=[],
        )

    with _override_view_dirs(tables_dir=tables_dir, figures_dir=figures_dir):
        table_paths = [
            views.write_resilience_profiles_csv(result, "resilience-profiles.csv"),
            views.write_resilience_stats_csv(result, "resilience-statistics.csv"),
        ]
        figure_paths = [
            views.plot_resilience_scatter(
                result,
                "resilience-scatter-threading.svg",
                variable="threading_ratio",
            ),
            views.plot_resilience_scatter(
                result,
                "resilience-scatter-depth.svg",
                variable="mean_depth",
            ),
            views.plot_resilience_boxplot(result, "resilience-boxplot.svg"),
            views.plot_resilience_indexed_trend(result, "resilience-indexed-trend.svg"),
        ]

    return AnalysisSection(
        title="Resilience",
        description="Engagement-vs-decline analysis across the GenAI cutoff.",
        table_paths=table_paths,
        figure_paths=figure_paths,
    )


@contextmanager
def _override_view_dirs(*, tables_dir: Path, figures_dir: Path) -> Iterator[None]:
    """Temporarily redirect ``views.py`` outputs to custom directories."""
    import views

    old_tables_dir = views.TABLES_DIR
    old_figures_dir = views.FIGURES_DIR
    views.TABLES_DIR = tables_dir
    views.FIGURES_DIR = figures_dir
    try:
        yield
    finally:
        views.TABLES_DIR = old_tables_dir
        views.FIGURES_DIR = old_figures_dir


def _build_summary_markdown(
    *,
    output_dir: Path,
    sections: list[AnalysisSection],
    wip_markdown_path: Path,
) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Analysis Summary",
        "",
        f"Generated: {timestamp}",
        "",
        "This file combines the filtered-data analysis outputs generated by `uv run python3 main.py all-analysis`.",
        "",
    ]

    if wip_markdown_path.exists():
        lines.append("## Key Results")
        lines.append(f"Source: [{wip_markdown_path.name}]({_relative_to_output(wip_markdown_path, output_dir)})")
        lines.append("")
        lines.extend(_wip_markdown_body(wip_markdown_path))
        lines.append("")

    lines.append("## Output Index")
    lines.append("")
    for section in sections:
        table_paths = _existing_paths(section.table_paths)
        figure_paths = _existing_paths(section.figure_paths)
        lines.append(f"### {section.title}")
        lines.append(section.description)
        lines.append("")

        if table_paths:
            lines.append("Tables")
            for path in table_paths:
                rel_path = _relative_to_output(path, output_dir)
                lines.append(f"- [{path.name}]({rel_path})")
            lines.append("")

        if figure_paths:
            lines.append("Figures")
            for path in figure_paths:
                rel_path = _relative_to_output(path, output_dir)
                lines.append(f"- [{path.name}]({rel_path})")
            lines.append("")
            for path in figure_paths:
                rel_path = _relative_to_output(path, output_dir)
                lines.append(f"![{path.stem}]({rel_path})")
                lines.append("")

        if not table_paths and not figure_paths:
            lines.append("No outputs were generated for this section.")
            lines.append("")

    return "\n".join(lines)


def _existing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _wip_markdown_body(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    if lines[0].startswith("# "):
        return lines[1:]
    return lines
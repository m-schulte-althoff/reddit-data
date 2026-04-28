"""Tests for src.all_analysis — filtered-data orchestration and summary output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import all_analysis as all_analysis_module


def test_all_analysis_writes_combined_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    def create_text(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def create_svg(path: Path) -> Path:
        return create_text(path, "<svg xmlns='http://www.w3.org/2000/svg'></svg>")

    original_describe = all_analysis_module.run_describe_outputs
    original_discursivity = all_analysis_module.run_discursivity_outputs
    original_helpers = all_analysis_module.run_helpers_outputs
    original_resilience = all_analysis_module.run_resilience_outputs
    original_ensure_panel = all_analysis_module.ensure_monthly_panel
    original_did = all_analysis_module.run_did_analysis
    original_resp = all_analysis_module.run_responsiveness_analysis
    original_mech = all_analysis_module.run_mechanisms_analysis
    original_ai = all_analysis_module.run_ai_mentions_analysis
    original_content = all_analysis_module.run_content_metrics_analysis
    original_interactions = all_analysis_module.run_interactions_analysis
    original_wip = all_analysis_module.run_wip_suite

    try:
        all_analysis_module.run_describe_outputs = lambda **_: all_analysis_module.AnalysisSection(
            title="Describe",
            description="Describe outputs.",
            table_paths=[create_text(tables_dir / "describe-comments-summary.csv", "a,b\n1,2\n")],
            figure_paths=[create_svg(figures_dir / "describe-comments-trend-aggregated.svg")],
        )
        all_analysis_module.run_discursivity_outputs = lambda **_: all_analysis_module.AnalysisSection(
            title="Discursivity",
            description="Discursivity outputs.",
            table_paths=[create_text(tables_dir / "discursivity-monthly.csv", "a,b\n1,2\n")],
            figure_paths=[create_svg(figures_dir / "discursivity-mean-depth-top15.svg")],
        )
        all_analysis_module.run_helpers_outputs = lambda **_: all_analysis_module.AnalysisSection(
            title="Helpers",
            description="Helper outputs.",
            table_paths=[create_text(tables_dir / "helpers-monthly.csv", "a,b\n1,2\n")],
            figure_paths=[create_svg(figures_dir / "helpers-gini-trend.svg")],
        )
        all_analysis_module.run_resilience_outputs = lambda **_: all_analysis_module.AnalysisSection(
            title="Resilience",
            description="Resilience outputs.",
            table_paths=[create_text(tables_dir / "resilience-statistics.csv", "a,b\n1,2\n")],
            figure_paths=[create_svg(figures_dir / "resilience-boxplot.svg")],
        )
        all_analysis_module.ensure_monthly_panel = lambda **_: (
            create_text(tables_dir / "community-monthly-panel.csv", "subreddit,month\naskreddit,2022-10\n"),
            create_text(tables_dir / "community-monthly-panel-metadata.json", "{}"),
            {"n_rows": 1},
        )
        all_analysis_module.run_did_analysis = lambda **_: type("Did", (), {
            "table_paths": {"summary": create_text(tables_dir / "did-summary.csv", "a,b\n1,2\n")},
            "figure_paths": {"event_comments": create_svg(figures_dir / "did-event-study-comments.svg")},
        })()
        all_analysis_module.run_responsiveness_analysis = lambda **_: type("Resp", (), {
            "table_paths": {"monthly": create_text(tables_dir / "responsiveness-monthly.csv", "a,b\n1,2\n")},
            "figure_paths": {"reply_rate": create_svg(figures_dir / "responsiveness-reply-rate-health-vs-general.svg")},
        })()
        all_analysis_module.run_mechanisms_analysis = lambda **_: type("Mech", (), {
            "table_paths": {"summary": create_text(tables_dir / "mechanism-moderation-summary.csv", "a,b\n1,2\n")},
            "figure_paths": {"coefficients": create_svg(figures_dir / "mechanism-moderation-coefficients.svg")},
        })()
        all_analysis_module.run_ai_mentions_analysis = lambda **_: type("Ai", (), {
            "table_paths": {"monthly": create_text(tables_dir / "ai-mentions-monthly.csv", "a,b\n1,2\n")},
            "figure_paths": {"comments": create_svg(figures_dir / "ai-mentions-health-vs-general-comments.svg")},
        })()
        all_analysis_module.run_content_metrics_analysis = lambda **_: type("Content", (), {
            "table_paths": {"monthly": create_text(tables_dir / "content-metrics-monthly.csv", "a,b\n1,2\n")},
            "figure_paths": {"length": create_svg(figures_dir / "content-length-health-vs-general.svg")},
        })()
        all_analysis_module.run_interactions_analysis = lambda **_: type("Interactions", (), {
            "table_paths": {"monthly": create_text(tables_dir / "interactions-monthly.csv", "a,b\n1,2\n")},
            "figure_paths": {"bond": create_svg(figures_dir / "interactions-bond-index-health-vs-general.svg")},
        })()
        all_analysis_module.run_wip_suite = lambda **_: type("Wip", (), {
            "summary_table": pd.DataFrame([{"section": "did", "metric": "comments_estimate", "group": "unweighted", "value": 0.1}]),
            "csv_path": create_text(tables_dir / "wip-key-results.csv", "a,b\n1,2\n"),
            "markdown_path": create_text(tables_dir / "wip-key-results.md", "# WIP Key Results\n\n- main finding\n"),
        })()

        artifacts = all_analysis_module.allAnalysis(output_dir=output_dir)
    finally:
        all_analysis_module.run_describe_outputs = original_describe
        all_analysis_module.run_discursivity_outputs = original_discursivity
        all_analysis_module.run_helpers_outputs = original_helpers
        all_analysis_module.run_resilience_outputs = original_resilience
        all_analysis_module.ensure_monthly_panel = original_ensure_panel
        all_analysis_module.run_did_analysis = original_did
        all_analysis_module.run_responsiveness_analysis = original_resp
        all_analysis_module.run_mechanisms_analysis = original_mech
        all_analysis_module.run_ai_mentions_analysis = original_ai
        all_analysis_module.run_content_metrics_analysis = original_content
        all_analysis_module.run_interactions_analysis = original_interactions
        all_analysis_module.run_wip_suite = original_wip

    assert artifacts.summary_path.exists()
    summary_text = artifacts.summary_path.read_text(encoding="utf-8")
    assert "# Analysis Summary" in summary_text
    assert "tables/community-monthly-panel.csv" in summary_text
    assert "figures/did-event-study-comments.svg" in summary_text
    assert "WIP Key Results" in summary_text
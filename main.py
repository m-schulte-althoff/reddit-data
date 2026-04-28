#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "libtorrent",
#     "orjson",
#     "zstandard",
#     "pandas",
# ]
# ///
"""CLI controller for the reddit-data pipeline.

Usage:
    uv run python3 main.py <command>

Commands:
    download       Download missing raw .zst files via Arctic Shift torrent.
    verify         Check that all raw files are present and valid.
    filter         Filter raw data to the configured time window.
    analyse        Compute descriptive statistics for comments and submissions.
    describe       Descriptive overview of filtered data (trends, per-subreddit).
    discursivity   Comment-depth / threading metrics from filtered data.
    resilience     Engagement vs. post-GenAI decline analysis.
    helpers        Repeat-helper concentration analysis.
    sample         Reservoir-sample records and write CSV + optional DataFrame.
    hf-extract     Extract data via Hugging Face (alternative source).
    hf-list        List months available on Hugging Face.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime

from src.config import LOGS_DIR, TABLES_DIR


def _setup_logging() -> None:
    """Configure root logger: stream + timestamped log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOGS_DIR / f"run_{ts}.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
    )


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_download() -> int:
    from src.arctic_shift import download

    download()
    return 0


def cmd_verify() -> int:
    from src.arctic_shift import verify

    ok = verify()
    return 0 if ok else 1


def cmd_filter() -> int:
    from src.arctic_shift import filter_raw

    filter_raw()
    return 0


def cmd_filter_subreddit() -> int:
    """Filter raw data by the Chan-2025 subreddit list.

    Supports CLI flags:
        --months 2022-06,2023-06   restrict to specific months
        --start  2022-06-01        inclusive start (ISO date)
        --end    2022-07-01        exclusive end (ISO date)
        --tag    chan2025           output filename tag
        --no-resume                start fresh instead of resuming
    """
    import argparse
    from datetime import datetime, timezone

    from src.filter import filter_all

    parser = argparse.ArgumentParser(prog="filter-subreddit")
    parser.add_argument("--months", type=str, default=None,
                        help="Comma-separated YYYY-MM months")
    parser.add_argument("--start", type=str, default=None,
                        help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Exclusive end date (YYYY-MM-DD)")
    parser.add_argument("--tag", type=str, default="chan2025",
                        help="Output filename tag")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume from previous run")

    # argv[0] is the script, argv[1] is the command name.
    args = parser.parse_args(sys.argv[2:])

    months = None
    if args.months:
        months = []
        for part in args.months.split(","):
            y, m = part.strip().split("-")
            months.append((int(y), int(m)))

    start_epoch = None
    if args.start:
        start_epoch = int(
            datetime.strptime(args.start, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    end_epoch = None
    if args.end:
        end_epoch = int(
            datetime.strptime(args.end, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    results = filter_all(
        output_tag=args.tag,
        months=months,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        resume=not args.no_resume,
    )
    for kind, stats in results.items():
        logging.getLogger(__name__).info(
            "%s: %d rows written → %s", kind, stats["rows_written"], stats["output_file"]
        )
    return 0


def cmd_analyse() -> int:
    from src.analysis import analyse
    from views import write_summary_csv

    for kind in ("comments", "submissions"):
        stats = analyse(kind)
        write_summary_csv(stats, f"analysis-{kind}-summary.csv")
        # Also dump full JSON for inspection.
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        json_out = TABLES_DIR / f"analysis-{kind}-summary.json"
        json_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.getLogger(__name__).info("Wrote %s", json_out)
    return 0


def cmd_describe() -> int:
    """Descriptive overview of filtered data: counts, trends, per-subreddit stats."""
    from src.describe import describe_filtered
    from views import (
        plot_describe_trend_aggregated,
        plot_describe_trend_by_community_type,
        plot_describe_trend_per_subreddit,
        write_describe_monthly_csv,
        write_describe_summary_csv,
    )

    for kind in ("comments", "submissions"):
        result = describe_filtered(kind)
        if result.total_records == 0:
            logging.getLogger(__name__).warning("No data for %s — skipping.", kind)
            continue
        write_describe_summary_csv(result, f"describe-{kind}-summary.csv")
        write_describe_monthly_csv(result, f"describe-{kind}-monthly.csv")
        write_describe_monthly_csv(result, f"describe-{kind}-monthly-top15.csv", top_n=15)
        plot_describe_trend_aggregated(result, f"describe-{kind}-trend-aggregated.svg")
        plot_describe_trend_by_community_type(
            result,
            f"describe-{kind}-trend-community-types.svg",
        )
        plot_describe_trend_per_subreddit(result, f"describe-{kind}-trend-all.svg", top_n=None)
        plot_describe_trend_per_subreddit(result, f"describe-{kind}-trend-top15.svg")
    return 0


def cmd_discursivity() -> int:
    """Compute comment-depth / discursivity metrics from filtered data."""
    from src.discursivity import (
        _discover_filtered_paths,
        compute_discursivity,
        save_discursivity,
    )
    from views import (
        plot_discursivity_mean_depth,
        plot_discursivity_threading_ratio,
        write_discursivity_csv,
    )

    result = compute_discursivity()
    if result.resolved_comments == 0:
        logging.getLogger(__name__).warning("No resolved comments — skipping outputs.")
        return 0

    # Persist result so downstream commands can reuse it.
    save_discursivity(
        result,
        comment_paths=_discover_filtered_paths("comments"),
        submission_paths=_discover_filtered_paths("submissions"),
    )

    write_discursivity_csv(result, "discursivity-monthly.csv")
    plot_discursivity_mean_depth(result, "discursivity-mean-depth-top15.svg")
    plot_discursivity_mean_depth(result, "discursivity-mean-depth-all.svg", top_n=None)
    plot_discursivity_threading_ratio(result, "discursivity-threading-ratio-top15.svg")
    plot_discursivity_threading_ratio(result, "discursivity-threading-ratio-all.svg", top_n=None)
    return 0


def cmd_resilience() -> int:
    """Engagement-vs-decline analysis across the GenAI cutoff."""
    from src.discursivity import (
        _discover_filtered_paths,
        compute_discursivity,
        load_discursivity,
        save_discursivity,
    )
    from src.resilience import compute_resilience
    from views import (
        plot_resilience_boxplot,
        plot_resilience_indexed_trend,
        plot_resilience_scatter,
        write_resilience_profiles_csv,
        write_resilience_stats_csv,
    )

    disc = load_discursivity()
    if disc is None:
        logging.getLogger(__name__).info("Recomputing discursivity (no valid cache).")
        disc = compute_discursivity()
        if disc.resolved_comments > 0:
            save_discursivity(
                disc,
                comment_paths=_discover_filtered_paths("comments"),
                submission_paths=_discover_filtered_paths("submissions"),
            )
    if disc.resolved_comments == 0:
        logging.getLogger(__name__).warning("No resolved comments — skipping.")
        return 0

    result = compute_resilience(disc)
    if not result.profiles:
        logging.getLogger(__name__).warning("No qualifying subreddits — skipping outputs.")
        return 0

    write_resilience_profiles_csv(result, "resilience-profiles.csv")
    write_resilience_stats_csv(result, "resilience-statistics.csv")
    plot_resilience_scatter(result, "resilience-scatter-threading.svg", variable="threading_ratio")
    plot_resilience_scatter(result, "resilience-scatter-depth.svg", variable="mean_depth")
    plot_resilience_boxplot(result, "resilience-boxplot.svg")
    plot_resilience_indexed_trend(result, "resilience-indexed-trend.svg")
    return 0


def cmd_sample() -> int:
    from src.analysis import sample
    from views import write_sample_csv

    for kind in ("comments", "submissions"):
        records = sample(kind)
        write_sample_csv(records, f"analysis-{kind}-sample.csv")
    return 0


def cmd_hf_extract() -> int:
    from src.hugging_face import extract

    extract()
    return 0


def cmd_hf_list() -> int:
    from src.hugging_face import list_available

    list_available()
    return 0


def cmd_helpers() -> int:
    from src.helpers import _discover_filtered_paths, analyse_helpers, compute_helpers
    from views import (
        plot_helpers_gini_trend,
        plot_helpers_moderation_scatter,
        plot_helpers_type_comparison,
        write_helpers_moderation_csv,
        write_helpers_monthly_csv,
        write_helpers_type_summary_csv,
    )

    result = compute_helpers(_discover_filtered_paths("comments"))
    if not result.cells:
        logging.getLogger(__name__).warning("No helper data — skipping.")
        return 0

    write_helpers_monthly_csv(result, "helpers-monthly.csv")

    analysis = analyse_helpers(result)
    write_helpers_type_summary_csv(analysis, "helpers-type-summary.csv")
    write_helpers_moderation_csv(analysis, "helpers-moderation.csv")
    plot_helpers_type_comparison(analysis, "helpers-type-comparison.svg")
    plot_helpers_moderation_scatter(analysis, "helpers-moderation-gini.svg", metric="gini")
    plot_helpers_moderation_scatter(analysis, "helpers-moderation-top5.svg", metric="top5_share")
    plot_helpers_gini_trend(result, "helpers-gini-trend.svg")
    return 0


COMMANDS: dict[str, tuple[callable, str]] = {  # type: ignore[type-arg]
    "download": (cmd_download, "Download missing raw .zst files via torrent"),
    "verify": (cmd_verify, "Check raw files are present and valid"),
    "filter": (cmd_filter, "Filter raw data to configured time window"),
    "filter-subreddit": (cmd_filter_subreddit, "Filter by subreddit list (Chan-2025)"),
    "analyse": (cmd_analyse, "Descriptive statistics for raw data"),
    "describe": (cmd_describe, "Descriptive overview of filtered data"),
    "discursivity": (cmd_discursivity, "Comment-depth / threading metrics"),
    "resilience": (cmd_resilience, "Engagement vs. post-GenAI decline analysis"),
    "helpers": (cmd_helpers, "Repeat-helper concentration analysis"),
    "sample": (cmd_sample, "Reservoir-sample records to CSV"),
    "hf-extract": (cmd_hf_extract, "Extract data via Hugging Face"),
    "hf-list": (cmd_hf_list, "List months available on Hugging Face"),
}


def main() -> int:
    _setup_logging()

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} <command>\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<14s} {desc}")
        return 2

    fn, _ = COMMANDS[sys.argv[1]]
    return fn()


if __name__ == "__main__":
    sys.exit(main())

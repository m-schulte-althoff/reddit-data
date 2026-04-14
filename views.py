"""Views — output formatting for tables, figures, and sample data."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.config import FIGURES_DIR, TABLES_DIR

# Maximum subreddits shown in a single line chart before it becomes unreadable.
_MAX_TREND_LINES: int = 50

if TYPE_CHECKING:
    from src.describe import DescribeResult

log = logging.getLogger(__name__)


def write_summary_csv(stats: dict, filename: str) -> Path:
    """Write a flat key-value summary CSV to output/tables/."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    flat: dict[str, str] = {}
    for k, v in stats.items():
        if isinstance(v, list):
            # e.g. top_subreddits_20 -> store as JSON string
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = str(v) if v is not None else ""

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in flat.items():
            writer.writerow([k, v])

    log.info("Wrote %s", out)
    return out


def write_sample_csv(records: list[dict], filename: str) -> Path:
    """Write sample records to a CSV in output/tables/."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    if not records:
        out.write_text("", encoding="utf-8")
        return out

    # Union of all keys across records, stable order.
    all_keys: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for k in rec:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    log.info("Wrote %s (%d rows)", out, len(records))
    return out


# ── Describe views ───────────────────────────────────────────────────────────


def write_describe_summary_csv(result: DescribeResult, filename: str) -> Path:
    """Write high-level summary stats to CSV."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename
    t_min, t_max = result.time_range()

    rows: list[tuple[str, str]] = [
        ("kind", result.kind),
        ("total_records", str(result.total_records)),
        ("unique_subreddits", str(len(result.subreddit_counts))),
        ("months_covered", str(len(result.monthly_counts))),
        ("time_range_start", t_min or ""),
        ("time_range_end", t_max or ""),
        ("parse_errors", str(result.parse_errors)),
    ]

    # Per-subreddit totals (sorted descending).
    for sub, count in result.subreddit_counts.most_common():
        rows.append((f"subreddit:{sub}", str(count)))

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    log.info("Wrote %s", out)
    return out


def write_describe_monthly_csv(
    result: DescribeResult,
    filename: str,
    top_n: int | None = None,
) -> Path:
    """Write a subreddit × month pivot table to CSV.

    When ``top_n`` is given, only the top subreddits by total volume are kept.
    The aggregated ``ALL`` row is always included.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    months = result.sorted_months()
    subreddits = _select_subreddits(result, top_n)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subreddit"] + months + ["total"])
        for sub in subreddits:
            row = [sub]
            total = 0
            for m in months:
                c = result.subreddit_monthly_counts.get((sub, m), 0)
                row.append(str(c))
                total += c
            row.append(str(total))
            writer.writerow(row)
        # Aggregated row.
        agg = ["ALL"]
        agg_total = 0
        for m in months:
            c = result.monthly_counts[m]
            agg.append(str(c))
            agg_total += c
        agg.append(str(agg_total))
        writer.writerow(agg)

    log.info("Wrote %s", out)
    return out


def plot_describe_trend_aggregated(result: DescribeResult, filename: str) -> Path:
    """Line chart of total posts per month (aggregated across subreddits)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    months = result.sorted_months()
    dates = _month_strings_to_dates(months)
    counts = [result.monthly_counts[m] for m in months]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, counts, marker="o", linewidth=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of posts")
    ax.set_title(f"Monthly post volume — {result.kind} (all subreddits)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _format_date_axis(ax, dates)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def plot_describe_trend_per_subreddit(
    result: DescribeResult,
    filename: str,
    top_n: int | None = 15,
) -> Path:
    """Line chart of monthly posts for all or the top *top_n* subreddits.

    When *top_n* is ``None`` all subreddits are included, but the plot is
    capped at ``_MAX_TREND_LINES`` for readability (full data is in the CSV).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    months = result.sorted_months()
    dates = _month_strings_to_dates(months)
    selected_subs = _select_subreddits(result, top_n)

    # Cap to a displayable number of lines.
    capped = False
    if len(selected_subs) > _MAX_TREND_LINES:
        log.warning(
            "Capping trend plot to top %d of %d subreddits (full data in CSV)",
            _MAX_TREND_LINES,
            len(selected_subs),
        )
        selected_subs = [
            sub for sub, _ in result.subreddit_counts.most_common(_MAX_TREND_LINES)
        ]
        capped = True

    use_markers = len(selected_subs) <= 15

    fig, ax = plt.subplots(figsize=(14, 7))
    for sub in selected_subs:
        counts = [result.subreddit_monthly_counts.get((sub, m), 0) for m in months]
        ax.plot(
            dates,
            counts,
            marker="." if use_markers else None,
            linewidth=1.0,
            label=sub,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of posts")
    if top_n is None and not capped:
        title_suffix = "all subreddits"
    elif capped:
        title_suffix = f"top {_MAX_TREND_LINES} of {len(result.subreddit_counts)} subreddits"
    else:
        title_suffix = f"top {top_n} subreddits"
    ax.set_title(f"Monthly post volume — {result.kind} ({title_suffix})")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _format_date_axis(ax, dates)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def _select_subreddits(result: DescribeResult, top_n: int | None) -> list[str]:
    """Return subreddit names ordered by total post count (descending)."""
    if top_n is None:
        return [sub for sub, _ in result.subreddit_counts.most_common()]
    return [sub for sub, _ in result.subreddit_counts.most_common(top_n)]


def _month_strings_to_dates(months: list[str]) -> list[datetime]:
    """Convert ``YYYY-MM`` strings to ``datetime`` objects for proper axis scaling."""
    return [datetime.strptime(m, "%Y-%m") for m in months]


def _format_date_axis(ax: plt.Axes, dates: list[datetime]) -> None:
    """Configure the x-axis as a date axis with readable tick spacing."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.figure.autofmt_xdate(rotation=45, ha="right")

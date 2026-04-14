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
    from src.discursivity import DepthBucket, DiscursivityResult
    from src.helpers import HelpersAnalysis, HelpersResult
    from src.resilience import ResilienceResult

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


# ── Discursivity views ───────────────────────────────────────────────────────


def write_discursivity_csv(result: DiscursivityResult, filename: str) -> Path:
    """Write subreddit × month discursivity metrics to a long-format CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    months = result.sorted_months()
    subreddits = result.sorted_subreddits()

    fieldnames = [
        "subreddit",
        "month",
        "comment_count",
        "submission_count",
        "mean_depth",
        "max_depth",
        "threading_ratio",
        "depth_1",
        "depth_2",
        "depth_3plus",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Per-subreddit rows.
        for sub in subreddits:
            for m in months:
                bucket = result.buckets.get((sub, m))
                sub_count = result.submission_counts.get((sub, m), 0)
                if bucket is None and sub_count == 0:
                    continue
                _write_discursivity_row(writer, sub, m, bucket, sub_count)

        # Aggregated ALL rows per month.
        for m in months:
            agg_bucket = _aggregate_month(result, m)
            agg_sub = sum(
                c for (_, mm), c in result.submission_counts.items() if mm == m
            )
            _write_discursivity_row(writer, "ALL", m, agg_bucket, agg_sub)

    log.info("Wrote %s", out)
    return out


def _write_discursivity_row(
    writer: csv.DictWriter,  # type: ignore[type-arg]
    sub: str,
    month: str,
    bucket: DepthBucket | None,
    sub_count: int,
) -> None:
    from src.discursivity import DepthBucket as _DepthBucket  # noqa: F811

    b = bucket or _DepthBucket()
    d3plus = sum(c for d, c in b.depth_histogram.items() if d >= 3)
    writer.writerow({
        "subreddit": sub,
        "month": month,
        "comment_count": b.count,
        "submission_count": sub_count,
        "mean_depth": round(b.mean_depth, 3),
        "max_depth": b.max_depth,
        "threading_ratio": round(b.threading_ratio, 3),
        "depth_1": b.depth_histogram.get(1, 0),
        "depth_2": b.depth_histogram.get(2, 0),
        "depth_3plus": d3plus,
    })


def _aggregate_month(
    result: DiscursivityResult,
    month: str,
) -> DepthBucket:
    """Combine all subreddit buckets for *month* into one."""
    from src.discursivity import DepthBucket as _DepthBucket  # noqa: F811

    agg = _DepthBucket()
    for (_, m), bucket in result.buckets.items():
        if m != month:
            continue
        agg.count += bucket.count
        agg.depth_sum += bucket.depth_sum
        if bucket.max_depth > agg.max_depth:
            agg.max_depth = bucket.max_depth
        agg.depth_histogram += bucket.depth_histogram
    return agg


def plot_discursivity_mean_depth(
    result: DiscursivityResult,
    filename: str,
    top_n: int | None = 15,
) -> Path:
    """Line chart of mean comment depth per month, per subreddit."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    months = result.sorted_months()
    dates = _month_strings_to_dates(months)
    subs = _pick_discursivity_subs(result, top_n)
    use_markers = len(subs) <= 15

    fig, ax = plt.subplots(figsize=(14, 7))
    for sub in subs:
        values = [
            result.buckets[(sub, m)].mean_depth
            if (sub, m) in result.buckets
            else 0.0
            for m in months
        ]
        ax.plot(
            dates,
            values,
            marker="." if use_markers else None,
            linewidth=1.0,
            label=sub,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Mean comment depth")
    ax.set_title(
        f"Mean comment depth over time ({_sub_label(top_n, len(result.sorted_subreddits()))})"
    )
    ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _format_date_axis(ax, dates)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def plot_discursivity_threading_ratio(
    result: DiscursivityResult,
    filename: str,
    top_n: int | None = 15,
) -> Path:
    """Line chart of threading ratio (% comments at depth >= 2) per month."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    months = result.sorted_months()
    dates = _month_strings_to_dates(months)
    subs = _pick_discursivity_subs(result, top_n)
    use_markers = len(subs) <= 15

    fig, ax = plt.subplots(figsize=(14, 7))
    for sub in subs:
        values = [
            result.buckets[(sub, m)].threading_ratio * 100
            if (sub, m) in result.buckets
            else 0.0
            for m in months
        ]
        ax.plot(
            dates,
            values,
            marker="." if use_markers else None,
            linewidth=1.0,
            label=sub,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Threading ratio (%)")
    ax.set_title(
        f"Threading ratio over time ({_sub_label(top_n, len(result.sorted_subreddits()))})"
    )
    ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _format_date_axis(ax, dates)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def _pick_discursivity_subs(
    result: DiscursivityResult,
    top_n: int | None,
) -> list[str]:
    """Select subreddits for discursivity plots (by comment volume)."""
    all_subs = result.sorted_subreddits()
    if top_n is None:
        return all_subs[:_MAX_TREND_LINES]
    return all_subs[:top_n]


def _sub_label(top_n: int | None, total: int) -> str:
    if top_n is None:
        shown = min(total, _MAX_TREND_LINES)
        if shown < total:
            return f"top {shown} of {total} subreddits"
        return "all subreddits"
    return f"top {top_n} subreddits"


# ── Resilience views ─────────────────────────────────────────────────────────


def write_resilience_profiles_csv(result: ResilienceResult, filename: str) -> Path:
    """Write per-subreddit pre/post profiles to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    fieldnames = [
        "subreddit",
        "pre_mean_comments",
        "post_mean_comments",
        "activity_change_pct",
        "pre_threading_ratio",
        "pre_mean_depth",
        "pre_months",
        "post_months",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in result.profiles:
            writer.writerow({
                "subreddit": p.subreddit,
                "pre_mean_comments": round(p.pre_mean_comments, 2),
                "post_mean_comments": round(p.post_mean_comments, 2),
                "activity_change_pct": round(p.activity_change_pct, 2),
                "pre_threading_ratio": round(p.pre_threading_ratio, 4),
                "pre_mean_depth": round(p.pre_mean_depth, 4),
                "pre_months": p.pre_months,
                "post_months": p.post_months,
            })

    log.info("Wrote %s", out)
    return out


def write_resilience_stats_csv(result: ResilienceResult, filename: str) -> Path:
    """Write statistical test results to a key-value CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    rows: list[tuple[str, str]] = [
        ("genai_cutoff", result.genai_cutoff),
        ("n_subreddits", str(len(result.profiles))),
    ]

    for label, corr in [
        ("threading_ratio", result.corr_threading),
        ("mean_depth", result.corr_depth),
    ]:
        if corr:
            rows.append((f"spearman_{label}_rho", f"{corr.rho:.4f}"))
            rows.append((f"spearman_{label}_p", f"{corr.p_value:.6f}"))
            rows.append((f"spearman_{label}_n", str(corr.n)))

    for label, reg in [
        ("threading_ratio", result.reg_threading),
        ("mean_depth", result.reg_depth),
    ]:
        if reg:
            rows.append((f"ols_{label}_slope", f"{reg.slope:.4f}"))
            rows.append((f"ols_{label}_intercept", f"{reg.intercept:.4f}"))
            rows.append((f"ols_{label}_r_squared", f"{reg.r_squared:.4f}"))
            rows.append((f"ols_{label}_p", f"{reg.p_value:.6f}"))

    for label, grp in [
        ("threading_ratio", result.group_threading),
        ("mean_depth", result.group_depth),
    ]:
        if grp:
            rows.append((f"mw_{label}_high_n", str(grp.high_n)))
            rows.append((f"mw_{label}_low_n", str(grp.low_n)))
            rows.append((f"mw_{label}_high_median_change", f"{grp.high_median_change:.2f}"))
            rows.append((f"mw_{label}_low_median_change", f"{grp.low_median_change:.2f}"))
            rows.append((f"mw_{label}_U", f"{grp.u_statistic:.0f}"))
            rows.append((f"mw_{label}_p", f"{grp.p_value:.6f}"))

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    log.info("Wrote %s", out)
    return out


def plot_resilience_scatter(
    result: ResilienceResult,
    filename: str,
    variable: str = "threading_ratio",
) -> Path:
    """Scatter plot of pre-period engagement vs. post-GenAI activity change.

    Includes an OLS regression line and a text box with Spearman rho,
    OLS R², and p-values.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    if variable == "threading_ratio":
        x = [p.pre_threading_ratio for p in result.profiles]
        xlabel = "Pre-GenAI threading ratio"
        corr = result.corr_threading
        reg = result.reg_threading
    else:
        x = [p.pre_mean_depth for p in result.profiles]
        xlabel = "Pre-GenAI mean comment depth"
        corr = result.corr_depth
        reg = result.reg_depth

    y = [p.activity_change_pct for p in result.profiles]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, alpha=0.6, s=30, edgecolors="k", linewidths=0.5)

    # OLS fit line.
    if reg and len(x) >= 2:
        x_min, x_max = min(x), max(x)
        margin = (x_max - x_min) * 0.02 or 0.01
        x_line = [x_min - margin + (x_max - x_min + 2 * margin) * i / 99 for i in range(100)]
        y_line = [reg.slope * xi + reg.intercept for xi in x_line]
        ax.plot(x_line, y_line, color="red", linewidth=1.5, linestyle="--", label="OLS fit")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    if corr and reg:
        text = (
            f"Spearman \u03c1 = {corr.rho:.3f} (p = {corr.p_value:.4f})\n"
            f"OLS R\u00b2 = {reg.r_squared:.3f}, slope = {reg.slope:.1f}\n"
            f"n = {corr.n} subreddits"
        )
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Activity change (%)")
    ax.set_title(f"Pre-GenAI {variable.replace('_', ' ')} vs. post-GenAI activity change")
    if reg:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def plot_resilience_boxplot(result: ResilienceResult, filename: str) -> Path:
    """Side-by-side box plots: activity change for high vs. low engagement groups."""
    import statistics as _stats  # needed for median split in view

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for ax, variable, label, grp in [
        (ax1, "threading_ratio", "Threading ratio", result.group_threading),
        (ax2, "mean_depth", "Mean depth", result.group_depth),
    ]:
        if variable == "threading_ratio":
            vals = [p.pre_threading_ratio for p in result.profiles]
        else:
            vals = [p.pre_mean_depth for p in result.profiles]

        med = _stats.median(vals)
        high = [p.activity_change_pct for p, v in zip(result.profiles, vals) if v >= med]
        low = [p.activity_change_pct for p, v in zip(result.profiles, vals) if v < med]

        bp = ax.boxplot(
            [low, high],
            tick_labels=[f"Low {label}\n(n={len(low)})", f"High {label}\n(n={len(high)})"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#bdd7ee")
        bp["boxes"][1].set_facecolor("#f8cbad")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Activity change (%)")

        if grp:
            text = f"MW U = {grp.u_statistic:.0f}\np = {grp.p_value:.4f}"
            ax.text(
                0.98, 0.98, text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
            )

    fig.suptitle("Post-GenAI activity change by pre-GenAI engagement level")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def plot_resilience_indexed_trend(result: ResilienceResult, filename: str) -> Path:
    """Indexed activity over time (pre-period mean = 100) for high/low groups."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    months = result.months
    dates = _month_strings_to_dates(months)
    high_vals = [result.indexed_high.get(m, 0.0) for m in months]
    low_vals = [result.indexed_low.get(m, 0.0) for m in months]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, high_vals, marker=".", linewidth=1.5, label="High threading (above median)")
    ax.plot(dates, low_vals, marker=".", linewidth=1.5, label="Low threading (below median)")

    # GenAI cutoff vertical line.
    cutoff_date = datetime.strptime(result.genai_cutoff, "%Y-%m")
    ax.axvline(
        cutoff_date, color="red", linewidth=1.5, linestyle="--",
        alpha=0.7, label="ChatGPT launch",
    )

    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Month")
    ax.set_ylabel("Indexed activity (pre-period mean = 100)")
    ax.set_title("Indexed comment activity: high vs. low engagement subreddits")
    ax.legend()
    _format_date_axis(ax, dates)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


# ── Helpers views ─────────────────────────────────────────────────────────────


def write_helpers_monthly_csv(result: HelpersResult, filename: str) -> Path:
    """Write per-subreddit per-month concentration metrics to long-format CSV."""
    from src.helpers import classify_subreddit

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    fieldnames = [
        "subreddit",
        "community_type",
        "month",
        "total_comments",
        "unique_authors",
        "top1_share",
        "top5_share",
        "hhi",
        "gini",
        "pct1_share",
        "pct9_share",
        "pct90_share",
    ]

    months = result.sorted_months()
    subreddits = result.sorted_subreddits()

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sub in subreddits:
            ctype = classify_subreddit(sub)
            for m in months:
                c = result.cells.get((sub, m))
                if c is None:
                    continue
                writer.writerow({
                    "subreddit": sub,
                    "community_type": ctype,
                    "month": m,
                    "total_comments": c.total_comments,
                    "unique_authors": c.unique_authors,
                    "top1_share": round(c.top1_share, 6),
                    "top5_share": round(c.top5_share, 6),
                    "hhi": round(c.hhi, 6),
                    "gini": round(c.gini, 4),
                    "pct1_share": round(c.pct1_share, 6),
                    "pct9_share": round(c.pct9_share, 6),
                    "pct90_share": round(c.pct90_share, 6),
                })

    log.info("Wrote %s", out)
    return out


def write_helpers_type_summary_csv(
    analysis: HelpersAnalysis,
    filename: str,
) -> Path:
    """Write community-type level summary to CSV."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    fieldnames = [
        "community_type",
        "n_subreddits",
        "mean_top1_share",
        "mean_top5_share",
        "mean_hhi",
        "mean_gini",
        "mean_pct1_share",
        "mean_pct9_share",
        "mean_pct90_share",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in analysis.type_summaries:
            writer.writerow({
                "community_type": s.community_type,
                "n_subreddits": s.n_subreddits,
                "mean_top1_share": round(s.mean_top1_share, 6),
                "mean_top5_share": round(s.mean_top5_share, 6),
                "mean_hhi": round(s.mean_hhi, 6),
                "mean_gini": round(s.mean_gini, 4),
                "mean_pct1_share": round(s.mean_pct1_share, 6),
                "mean_pct9_share": round(s.mean_pct9_share, 6),
                "mean_pct90_share": round(s.mean_pct90_share, 6),
            })

    log.info("Wrote %s", out)
    return out


def write_helpers_moderation_csv(
    analysis: HelpersAnalysis,
    filename: str,
) -> Path:
    """Write per-subreddit moderation (concentration vs. activity change)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / filename

    fieldnames = [
        "subreddit",
        "community_type",
        "mean_gini",
        "mean_top1_share",
        "mean_top5_share",
        "mean_hhi",
        "mean_pct1_share",
        "total_comments_first_half",
        "total_comments_second_half",
        "activity_change_pct",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in analysis.moderation_rows:
            writer.writerow({
                "subreddit": r.subreddit,
                "community_type": r.community_type,
                "mean_gini": round(r.mean_gini, 4),
                "mean_top1_share": round(r.mean_top1_share, 6),
                "mean_top5_share": round(r.mean_top5_share, 6),
                "mean_hhi": round(r.mean_hhi, 6),
                "mean_pct1_share": round(r.mean_pct1_share, 6),
                "total_comments_first_half": r.total_comments_first_half,
                "total_comments_second_half": r.total_comments_second_half,
                "activity_change_pct": round(r.activity_change_pct, 2),
            })

    log.info("Wrote %s", out)
    return out


def plot_helpers_type_comparison(
    analysis: HelpersAnalysis,
    filename: str,
) -> Path:
    """Grouped bar chart comparing concentration metrics by community type."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    import numpy as np

    metrics = ["top1_share", "top5_share", "gini", "pct1_share"]
    labels = ["Top-1 share", "Top-5 share", "Gini", "Top 1 % share"]

    type_data: dict[str, list[float]] = {}
    for s in analysis.type_summaries:
        type_data[s.community_type] = [
            s.mean_top1_share,
            s.mean_top5_share,
            s.mean_gini,
            s.mean_pct1_share,
        ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    colors = {"general": "#5b9bd5", "health": "#ed7d31"}

    for i, ctype in enumerate(("general", "health")):
        vals = type_data.get(ctype, [0] * len(metrics))
        ax.bar(x + i * width, vals, width, label=ctype.capitalize(), color=colors[ctype])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean value")
    ax.set_title("Helper concentration: General vs. Health communities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def plot_helpers_moderation_scatter(
    analysis: HelpersAnalysis,
    filename: str,
    metric: str = "gini",
) -> Path:
    """Scatter: concentration metric vs. activity change, coloured by type."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    label_map = {
        "gini": ("Mean Gini coefficient", "mean_gini"),
        "top1_share": ("Mean top-1 share", "mean_top1_share"),
        "top5_share": ("Mean top-5 share", "mean_top5_share"),
        "hhi": ("Mean HHI", "mean_hhi"),
        "pct1_share": ("Top 1 % user share", "mean_pct1_share"),
    }
    xlabel, attr = label_map.get(metric, ("Metric", metric))
    colors = {"general": "#5b9bd5", "health": "#ed7d31"}

    fig, ax = plt.subplots(figsize=(10, 7))

    for ctype in ("general", "health"):
        rows = [r for r in analysis.moderation_rows if r.community_type == ctype]
        x = [getattr(r, attr) for r in rows]
        y = [r.activity_change_pct for r in rows]
        ax.scatter(
            x, y,
            label=ctype.capitalize(),
            color=colors[ctype],
            alpha=0.7,
            s=40,
            edgecolors="k",
            linewidths=0.5,
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Activity change (%)")
    ax.set_title(f"Helper concentration ({xlabel}) vs. activity change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out


def plot_helpers_gini_trend(
    result: HelpersResult,
    filename: str,
    top_n: int | None = 15,
) -> Path:
    """Line chart of monthly Gini coefficient for top subreddits."""
    from src.helpers import classify_subreddit

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename

    months = result.sorted_months()
    dates = _month_strings_to_dates(months)
    subs = result.sorted_subreddits()
    if top_n is not None:
        subs = subs[:top_n]
    else:
        subs = subs[:_MAX_TREND_LINES]
    use_markers = len(subs) <= 15

    fig, ax = plt.subplots(figsize=(14, 7))
    for sub in subs:
        vals = [
            result.cells[(sub, m)].gini
            if (sub, m) in result.cells
            else 0.0
            for m in months
        ]
        ctype = classify_subreddit(sub)
        ls = "--" if ctype == "general" else "-"
        ax.plot(
            dates,
            vals,
            marker="." if use_markers else None,
            linewidth=1.0,
            linestyle=ls,
            label=f"{sub} ({ctype[0].upper()})",
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Helper concentration (Gini) over time")
    ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _format_date_axis(ax, dates)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    log.info("Wrote %s", out)
    return out

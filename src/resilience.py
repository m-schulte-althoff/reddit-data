"""Resilience analysis — engagement depth vs. post-GenAI activity decline.

Tests whether subreddits with deeper, more threaded discussions (higher
threading ratio / mean comment depth) experienced a smaller decline in
monthly comment volume after the ChatGPT launch (November 2022).

Exploratory outputs:

- Scatter plots with OLS fit lines (threading ratio / depth vs. change)
- Indexed activity trend for high- vs. low-engagement groups
- Box plot comparing decline distributions by engagement group

Rigorous tests:

- Spearman rank correlation (non-parametric)
- OLS regression (parametric effect size)
- Mann–Whitney U test (non-parametric group comparison)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

from scipy import stats as sp_stats

from src.discursivity import DiscursivityResult

log = logging.getLogger(__name__)

GENAI_CUTOFF: str = "2022-11"


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class SubredditProfile:
    """Pre/post-GenAI engagement summary for one subreddit."""

    subreddit: str
    pre_mean_comments: float
    post_mean_comments: float
    activity_change_pct: float  # (post − pre) / pre × 100
    pre_threading_ratio: float  # comment-count weighted mean
    pre_mean_depth: float  # comment-count weighted mean
    pre_months: int  # months with >0 comments in pre-period
    post_months: int  # months with >0 comments in post-period


@dataclass
class CorrelationResult:
    """Spearman rank-correlation output."""

    variable: str
    rho: float
    p_value: float
    n: int


@dataclass
class RegressionResult:
    """Simple OLS: activity_change ~ predictor."""

    variable: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n: int


@dataclass
class GroupComparison:
    """Mann–Whitney U group comparison (median split)."""

    variable: str
    high_n: int
    low_n: int
    high_median_change: float
    low_median_change: float
    u_statistic: float
    p_value: float


@dataclass
class ResilienceResult:
    """Full output of the resilience analysis."""

    genai_cutoff: str = GENAI_CUTOFF
    profiles: list[SubredditProfile] = field(default_factory=list)
    months: list[str] = field(default_factory=list)

    # Indexed trends — median split by pre-period threading ratio.
    indexed_high: dict[str, float] = field(default_factory=dict)
    indexed_low: dict[str, float] = field(default_factory=dict)

    # Statistical results (``None`` when too few qualifying subreddits).
    corr_threading: CorrelationResult | None = None
    corr_depth: CorrelationResult | None = None
    reg_threading: RegressionResult | None = None
    reg_depth: RegressionResult | None = None
    group_threading: GroupComparison | None = None
    group_depth: GroupComparison | None = None


# ── Core engine ──────────────────────────────────────────────────────────────


def compute_resilience(
    disc: DiscursivityResult,
    *,
    genai_cutoff: str = GENAI_CUTOFF,
    min_pre_months: int = 3,
    min_post_months: int = 3,
    min_comments_per_month: float = 10.0,
) -> ResilienceResult:
    """Compute engagement-vs-decline analysis across the GenAI cutoff.

    Parameters
    ----------
    disc
        Output of :func:`~src.discursivity.compute_discursivity`.
    genai_cutoff
        ``YYYY-MM`` month string marking the GenAI era start.
    min_pre_months
        Minimum pre-cutoff months with >0 comments for a subreddit to qualify.
    min_post_months
        Minimum post-cutoff months with >0 comments.
    min_comments_per_month
        Minimum pre-period mean monthly comment count.
    """
    result = ResilienceResult(genai_cutoff=genai_cutoff)
    all_months = disc.sorted_months()
    result.months = all_months

    pre_months = [m for m in all_months if m < genai_cutoff]
    post_months = [m for m in all_months if m >= genai_cutoff]

    if not pre_months or not post_months:
        log.warning("No data on both sides of cutoff %s", genai_cutoff)
        return result

    # ── Build per-subreddit profiles ─────────────────────────────────────
    for sub in disc.sorted_subreddits():
        profile = _build_profile(disc, sub, pre_months, post_months)
        if profile is None:
            continue
        if profile.pre_months < min_pre_months:
            continue
        if profile.post_months < min_post_months:
            continue
        if profile.pre_mean_comments < min_comments_per_month:
            continue
        result.profiles.append(profile)

    # Stable sort for deterministic output.
    result.profiles.sort(key=lambda p: p.subreddit)

    n = len(result.profiles)
    log.info("%d subreddits qualify for resilience analysis", n)

    if n < 5:
        log.warning("Too few subreddits (%d) for statistical tests", n)
        return result

    # ── Statistical tests ────────────────────────────────────────────────
    changes = [p.activity_change_pct for p in result.profiles]
    tr = [p.pre_threading_ratio for p in result.profiles]
    md = [p.pre_mean_depth for p in result.profiles]

    result.corr_threading = _spearman(tr, changes, "threading_ratio")
    result.corr_depth = _spearman(md, changes, "mean_depth")
    result.reg_threading = _ols(tr, changes, "threading_ratio")
    result.reg_depth = _ols(md, changes, "mean_depth")
    result.group_threading = _group_comparison(result.profiles, "threading_ratio")
    result.group_depth = _group_comparison(result.profiles, "mean_depth")

    # ── Indexed trends ───────────────────────────────────────────────────
    _compute_indexed_trends(disc, result)

    log.info(
        "Resilience: Spearman rho(threading)=%.3f (p=%.4f), rho(depth)=%.3f (p=%.4f)",
        result.corr_threading.rho if result.corr_threading else 0,
        result.corr_threading.p_value if result.corr_threading else 1,
        result.corr_depth.rho if result.corr_depth else 0,
        result.corr_depth.p_value if result.corr_depth else 1,
    )
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_profile(
    disc: DiscursivityResult,
    sub: str,
    pre_months: list[str],
    post_months: list[str],
) -> SubredditProfile | None:
    """Build a single subreddit profile from the discursivity buckets."""
    # Pre-period.
    pre_counts: list[int] = []
    pre_tr_w, pre_md_w, pre_total = 0.0, 0.0, 0
    active_pre = 0
    for m in pre_months:
        b = disc.buckets.get((sub, m))
        c = b.count if b else 0
        pre_counts.append(c)
        if b and b.count > 0:
            active_pre += 1
            pre_tr_w += b.threading_ratio * b.count
            pre_md_w += b.mean_depth * b.count
            pre_total += b.count

    # Post-period.
    post_counts: list[int] = []
    active_post = 0
    for m in post_months:
        b = disc.buckets.get((sub, m))
        c = b.count if b else 0
        post_counts.append(c)
        if c > 0:
            active_post += 1

    pre_mean = statistics.mean(pre_counts) if pre_counts else 0.0
    post_mean = statistics.mean(post_counts) if post_counts else 0.0

    if pre_mean == 0:
        return None

    return SubredditProfile(
        subreddit=sub,
        pre_mean_comments=pre_mean,
        post_mean_comments=post_mean,
        activity_change_pct=(post_mean - pre_mean) / pre_mean * 100,
        pre_threading_ratio=pre_tr_w / pre_total if pre_total else 0.0,
        pre_mean_depth=pre_md_w / pre_total if pre_total else 0.0,
        pre_months=active_pre,
        post_months=active_post,
    )


def _spearman(x: list[float], y: list[float], name: str) -> CorrelationResult:
    """Spearman rank correlation between *x* and *y*."""
    rho, p = sp_stats.spearmanr(x, y)
    return CorrelationResult(
        variable=name, rho=float(rho), p_value=float(p), n=len(x),
    )


def _ols(x: list[float], y: list[float], name: str) -> RegressionResult:
    """Simple OLS regression: y ~ x."""
    res = sp_stats.linregress(x, y)
    return RegressionResult(
        variable=name,
        slope=float(res.slope),
        intercept=float(res.intercept),
        r_squared=float(res.rvalue ** 2),
        p_value=float(res.pvalue),
        n=len(x),
    )


def _group_comparison(
    profiles: list[SubredditProfile],
    variable: str,
) -> GroupComparison | None:
    """Median-split Mann–Whitney U comparison."""
    if variable == "threading_ratio":
        vals = [p.pre_threading_ratio for p in profiles]
    else:
        vals = [p.pre_mean_depth for p in profiles]

    med = statistics.median(vals)
    high = [p.activity_change_pct for p, v in zip(profiles, vals) if v >= med]
    low = [p.activity_change_pct for p, v in zip(profiles, vals) if v < med]

    # All values equal → force split at midpoint.
    if not high or not low:
        sorted_pairs = sorted(zip(vals, profiles), key=lambda t: t[0])
        mid = len(sorted_pairs) // 2
        low = [p.activity_change_pct for _, p in sorted_pairs[:mid]]
        high = [p.activity_change_pct for _, p in sorted_pairs[mid:]]

    if not high or not low:
        return None

    u, p = sp_stats.mannwhitneyu(high, low, alternative="two-sided")
    return GroupComparison(
        variable=variable,
        high_n=len(high),
        low_n=len(low),
        high_median_change=statistics.median(high),
        low_median_change=statistics.median(low),
        u_statistic=float(u),
        p_value=float(p),
    )


def _compute_indexed_trends(
    disc: DiscursivityResult,
    result: ResilienceResult,
) -> None:
    """Indexed activity (pre-period mean = 100) for high/low engagement groups."""
    tr_values = [p.pre_threading_ratio for p in result.profiles]
    med_tr = statistics.median(tr_values)

    high_subs = {
        p.subreddit for p in result.profiles if p.pre_threading_ratio >= med_tr
    }
    low_subs = {
        p.subreddit for p in result.profiles if p.pre_threading_ratio < med_tr
    }

    # Fallback: force split if median puts everything in one group.
    if not low_subs:
        sorted_p = sorted(result.profiles, key=lambda p: p.pre_threading_ratio)
        mid = len(sorted_p) // 2
        low_subs = {p.subreddit for p in sorted_p[:mid]}
        high_subs = {p.subreddit for p in sorted_p[mid:]}

    pre_means = {p.subreddit: p.pre_mean_comments for p in result.profiles}

    for month in result.months:
        h_idx: list[float] = []
        l_idx: list[float] = []
        for p in result.profiles:
            b = disc.buckets.get((p.subreddit, month))
            c = b.count if b else 0
            indexed = (c / pre_means[p.subreddit]) * 100 if pre_means[p.subreddit] > 0 else 0.0
            if p.subreddit in high_subs:
                h_idx.append(indexed)
            else:
                l_idx.append(indexed)

        result.indexed_high[month] = statistics.mean(h_idx) if h_idx else 0.0
        result.indexed_low[month] = statistics.mean(l_idx) if l_idx else 0.0

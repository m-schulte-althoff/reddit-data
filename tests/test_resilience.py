"""Tests for src/resilience.py — engagement vs. post-GenAI decline analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.discursivity import DepthBucket, DiscursivityResult
from src.resilience import (
    GENAI_CUTOFF,
    ResilienceResult,
    SubredditProfile,
    _build_profile,
    _group_comparison,
    _ols,
    _spearman,
    compute_resilience,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _bucket_with(depths: dict[int, int]) -> DepthBucket:
    """Create a DepthBucket from a {depth: count} mapping."""
    b = DepthBucket()
    for depth, count in depths.items():
        for _ in range(count):
            b.add(depth)
    return b


def _make_disc(
    subreddits: dict[str, dict[str, dict[int, int]]],
) -> DiscursivityResult:
    """Build a DiscursivityResult from nested {sub: {month: {depth: count}}}.

    Also sets total/resolved comment counts consistently.
    """
    result = DiscursivityResult()
    total = 0
    for sub, months in subreddits.items():
        for month, depths in months.items():
            b = _bucket_with(depths)
            result.buckets[(sub, month)] = b
            total += b.count
    result.total_comments = total
    result.resolved_comments = total
    return result


# Six subreddits spanning 2022-06 to 2023-04:
#   pre-period:  2022-06 .. 2022-10  (5 months)
#   post-period: 2022-11 .. 2023-04  (6 months)
#
# Designed so that higher-threading subs have less decline.
DISC_DATA: dict[str, dict[str, dict[int, int]]] = {
    "sub_a": {
        # High threading (0.7), mild decline.
        **{m: {1: 30, 2: 50, 3: 20} for m in ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]},
        **{m: {1: 24, 2: 40, 3: 16} for m in ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]},
    },
    "sub_b": {
        # Threading ~0.6, moderate decline.
        **{m: {1: 40, 2: 40, 3: 20} for m in ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]},
        **{m: {1: 24, 2: 24, 3: 12} for m in ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]},
    },
    "sub_c": {
        # Threading ~0.5, moderate decline.
        **{m: {1: 50, 2: 30, 3: 20} for m in ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]},
        **{m: {1: 25, 2: 15, 3: 10} for m in ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]},
    },
    "sub_d": {
        # Threading ~0.3, large decline.
        **{m: {1: 70, 2: 20, 3: 10} for m in ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]},
        **{m: {1: 21, 2: 6, 3: 3} for m in ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]},
    },
    "sub_e": {
        # Threading ~0.2, large decline.
        **{m: {1: 80, 2: 15, 3: 5} for m in ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]},
        **{m: {1: 16, 2: 3, 3: 1} for m in ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]},
    },
    "sub_f": {
        # Tiny subreddit (should be filtered out at default thresholds).
        **{m: {1: 2, 2: 1} for m in ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]},
        **{m: {1: 1} for m in ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]},
    },
}


@pytest.fixture()
def disc() -> DiscursivityResult:
    return _make_disc(DISC_DATA)


# ── SubredditProfile / _build_profile ────────────────────────────────────────


def test_build_profile_basic(disc: DiscursivityResult) -> None:
    pre = ["2022-06", "2022-07", "2022-08", "2022-09", "2022-10"]
    post = ["2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04"]

    p = _build_profile(disc, "sub_a", pre, post)
    assert p is not None
    assert p.subreddit == "sub_a"
    assert p.pre_months == 5
    assert p.post_months == 6
    assert p.pre_mean_comments == 100.0  # 100 per pre-month
    assert p.post_mean_comments == 80.0  # 80 per post-month
    assert p.activity_change_pct == pytest.approx(-20.0)
    # threading = 70/100 = 0.7
    assert p.pre_threading_ratio == pytest.approx(0.7)


def test_build_profile_returns_none_for_zero_pre(disc: DiscursivityResult) -> None:
    pre = ["2020-01", "2020-02", "2020-03"]  # no data at all
    post = ["2022-11", "2022-12", "2023-01"]
    assert _build_profile(disc, "sub_a", pre, post) is None


# ── compute_resilience integration ───────────────────────────────────────────


def test_compute_resilience_basic(disc: DiscursivityResult) -> None:
    result = compute_resilience(disc, min_comments_per_month=10.0)

    # sub_f should be filtered out (pre_mean = 3 comments < 10).
    subs = [p.subreddit for p in result.profiles]
    assert "sub_f" not in subs
    assert len(result.profiles) == 5  # a, b, c, d, e


def test_compute_resilience_profiles_sorted(disc: DiscursivityResult) -> None:
    result = compute_resilience(disc, min_comments_per_month=10.0)
    names = [p.subreddit for p in result.profiles]
    assert names == sorted(names)


def test_compute_resilience_decline_order(disc: DiscursivityResult) -> None:
    """Higher threading subs should have less decline in our test data."""
    result = compute_resilience(disc, min_comments_per_month=10.0)
    by_change = sorted(result.profiles, key=lambda p: p.activity_change_pct, reverse=True)
    # sub_a (most threading) should have the smallest decline (largest change).
    assert by_change[0].subreddit == "sub_a"


def test_compute_resilience_stats_present(disc: DiscursivityResult) -> None:
    result = compute_resilience(disc, min_comments_per_month=10.0)
    assert result.corr_threading is not None
    assert result.corr_depth is not None
    assert result.reg_threading is not None
    assert result.reg_depth is not None
    assert result.group_threading is not None
    assert result.group_depth is not None


def test_compute_resilience_spearman_positive(disc: DiscursivityResult) -> None:
    """Higher threading → less decline → positive Spearman rho."""
    result = compute_resilience(disc, min_comments_per_month=10.0)
    assert result.corr_threading is not None
    assert result.corr_threading.rho > 0


def test_compute_resilience_too_few_skips_stats(disc: DiscursivityResult) -> None:
    """With very high min_comments threshold, too few subs qualify → no stats."""
    result = compute_resilience(disc, min_comments_per_month=200.0)
    assert result.corr_threading is None


def test_compute_resilience_indexed_trends(disc: DiscursivityResult) -> None:
    result = compute_resilience(disc, min_comments_per_month=10.0)
    assert len(result.indexed_high) > 0
    assert len(result.indexed_low) > 0
    # Pre-period months should average near 100.
    pre_high = [result.indexed_high[m] for m in result.months if m < GENAI_CUTOFF]
    assert all(v > 50 for v in pre_high)  # sanity: shouldn't be near zero


def test_compute_resilience_no_pre_data() -> None:
    """All data is post-cutoff → empty result."""
    disc = _make_disc({
        "sub_x": {m: {1: 50, 2: 50} for m in ["2023-01", "2023-02", "2023-03"]},
    })
    result = compute_resilience(disc)
    assert len(result.profiles) == 0


def test_compute_resilience_custom_cutoff(disc: DiscursivityResult) -> None:
    """Using a custom cutoff shifts the split."""
    result = compute_resilience(disc, genai_cutoff="2022-08")
    assert result.genai_cutoff == "2022-08"
    # Fewer pre-months may filter some subs out (only 2 pre months),
    # but the computation should succeed.
    assert isinstance(result, ResilienceResult)


# ── Statistical helpers ──────────────────────────────────────────────────────


def test_spearman_basic() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]  # perfect rank correlation
    c = _spearman(x, y, "test")
    assert c.rho == pytest.approx(1.0)
    assert c.p_value < 0.05
    assert c.n == 5


def test_ols_basic() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]  # y = 2x + 1
    r = _ols(x, y, "test")
    assert r.slope == pytest.approx(2.0)
    assert r.intercept == pytest.approx(1.0)
    assert r.r_squared == pytest.approx(1.0)
    assert r.n == 5


def test_group_comparison_basic() -> None:
    profiles = [
        SubredditProfile("a", 100, 90, -10, 0.8, 1.5, 5, 5),
        SubredditProfile("b", 100, 85, -15, 0.7, 1.4, 5, 5),
        SubredditProfile("c", 100, 80, -20, 0.6, 1.3, 5, 5),
        SubredditProfile("d", 100, 50, -50, 0.3, 1.1, 5, 5),
        SubredditProfile("e", 100, 40, -60, 0.2, 1.0, 5, 5),
        SubredditProfile("f", 100, 30, -70, 0.1, 0.9, 5, 5),
    ]
    grp = _group_comparison(profiles, "threading_ratio")
    assert grp is not None
    assert grp.high_n > 0
    assert grp.low_n > 0
    # High group should have less decline (higher median change).
    assert grp.high_median_change > grp.low_median_change


# ── View tests ───────────────────────────────────────────────────────────────


def test_write_resilience_profiles_csv(
    disc: DiscursivityResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import write_resilience_profiles_csv

    monkeypatch.setattr("src.config.TABLES_DIR", tmp_path)
    result = compute_resilience(disc, min_comments_per_month=10.0)
    out = write_resilience_profiles_csv(result, "test-profiles.csv")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "subreddit" in text
    assert "activity_change_pct" in text
    assert "sub_a" in text


def test_write_resilience_stats_csv(
    disc: DiscursivityResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import write_resilience_stats_csv

    monkeypatch.setattr("src.config.TABLES_DIR", tmp_path)
    result = compute_resilience(disc, min_comments_per_month=10.0)
    out = write_resilience_stats_csv(result, "test-stats.csv")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "spearman_threading_ratio_rho" in text
    assert "ols_threading_ratio_slope" in text
    assert "mw_threading_ratio_p" in text


def test_plot_resilience_scatter(
    disc: DiscursivityResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import plot_resilience_scatter

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    result = compute_resilience(disc, min_comments_per_month=10.0)
    out = plot_resilience_scatter(result, "test-scatter.svg", variable="threading_ratio")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_resilience_boxplot(
    disc: DiscursivityResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import plot_resilience_boxplot

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    result = compute_resilience(disc, min_comments_per_month=10.0)
    out = plot_resilience_boxplot(result, "test-boxplot.svg")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_resilience_indexed_trend(
    disc: DiscursivityResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from views import plot_resilience_indexed_trend

    monkeypatch.setattr("src.config.FIGURES_DIR", tmp_path)
    result = compute_resilience(disc, min_comments_per_month=10.0)
    out = plot_resilience_indexed_trend(result, "test-indexed.svg")
    assert out.exists()
    assert out.stat().st_size > 0

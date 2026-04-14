"""Tests for src.helpers — repeat-helper concentration analysis."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.helpers import (
    ConcentrationMetrics,
    HelpersResult,
    _compute_metrics,
    _gini,
    analyse_helpers,
    classify_subreddit,
    compute_helpers,
)
from tests.conftest import write_zst_jsonl


# ── classify_subreddit ───────────────────────────────────────────────────────


class TestClassifySubreddit:
    def test_general(self):
        assert classify_subreddit("AskReddit") == "general"
        assert classify_subreddit("Fitness") == "general"

    def test_health(self):
        assert classify_subreddit("Depression") == "health"
        assert classify_subreddit("ADHD") == "health"

    def test_other(self):
        assert classify_subreddit("funny") == "other"
        assert classify_subreddit("NonExistent") == "other"


# ── _gini ────────────────────────────────────────────────────────────────────


class TestGini:
    def test_empty(self):
        assert _gini([]) == 0.0

    def test_all_zeros(self):
        assert _gini([0, 0, 0]) == 0.0

    def test_perfect_equality(self):
        assert _gini([10, 10, 10, 10]) == pytest.approx(0.0)

    def test_maximal_inequality(self):
        # One person has everything.
        g = _gini([0, 0, 0, 100])
        assert g > 0.7

    def test_single_value(self):
        assert _gini([42]) == pytest.approx(0.0)


# ── _compute_metrics ─────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_empty_counter(self):
        m = _compute_metrics(Counter())
        assert m.total_comments == 0
        assert m.unique_authors == 0

    def test_single_author(self):
        m = _compute_metrics(Counter({"alice": 10}))
        assert m.total_comments == 10
        assert m.unique_authors == 1
        assert m.top1_share == pytest.approx(1.0)
        assert m.top5_share == pytest.approx(1.0)
        assert m.hhi == pytest.approx(1.0)
        assert m.gini == pytest.approx(0.0)
        # With 1 author: top 1 % includes 1 person.
        assert m.pct1_share == pytest.approx(1.0)

    def test_two_equal_authors(self):
        m = _compute_metrics(Counter({"alice": 5, "bob": 5}))
        assert m.total_comments == 10
        assert m.unique_authors == 2
        assert m.top1_share == pytest.approx(0.5)
        assert m.top5_share == pytest.approx(1.0)
        assert m.hhi == pytest.approx(0.5)
        assert m.gini == pytest.approx(0.0, abs=0.01)

    def test_skewed_distribution(self):
        counts = Counter({"power": 90, "mid": 5, "low1": 3, "low2": 2})
        m = _compute_metrics(counts)
        assert m.total_comments == 100
        assert m.unique_authors == 4
        assert m.top1_share == pytest.approx(0.9)
        assert m.top5_share == pytest.approx(1.0)
        assert m.hhi > 0.5  # Dominated by one user.
        assert m.gini > 0.5  # Highly unequal.

    def test_pct_shares_sum_to_one(self):
        counts = Counter({f"user{i}": i + 1 for i in range(100)})
        m = _compute_metrics(counts)
        total_share = m.pct1_share + m.pct9_share + m.pct90_share
        assert total_share == pytest.approx(1.0, abs=1e-9)


# ── HelpersResult ────────────────────────────────────────────────────────────


class TestHelpersResult:
    def test_sorted_months(self):
        r = HelpersResult(cells={
            ("sub", "2023-03"): ConcentrationMetrics(total_comments=1),
            ("sub", "2023-01"): ConcentrationMetrics(total_comments=2),
            ("sub", "2023-02"): ConcentrationMetrics(total_comments=3),
        })
        assert r.sorted_months() == ["2023-01", "2023-02", "2023-03"]

    def test_sorted_subreddits(self):
        r = HelpersResult(cells={
            ("bigSub", "2023-01"): ConcentrationMetrics(total_comments=100),
            ("smallSub", "2023-01"): ConcentrationMetrics(total_comments=10),
        })
        subs = r.sorted_subreddits()
        assert subs[0] == "bigSub"
        assert subs[1] == "smallSub"


# ── compute_helpers integration ──────────────────────────────────────────────


def _make_comments(
    subreddit: str,
    authors_counts: dict[str, int],
    ts: int = 1654100000,
) -> list[dict]:
    """Build comment records for test data."""
    records = []
    idx = 0
    for author, count in authors_counts.items():
        for _ in range(count):
            records.append({
                "id": f"c_{subreddit}_{idx}",
                "body": "test",
                "subreddit": subreddit,
                "author": author,
                "created_utc": ts,
            })
            idx += 1
    return records


class TestComputeHelpers:
    def test_basic(self, tmp_path: Path):
        records = _make_comments("askreddit", {"alice": 8, "bob": 2})
        p = tmp_path / "test.jsonl.zst"
        write_zst_jsonl(p, records)

        result = compute_helpers([p])
        assert len(result.cells) == 1
        key = ("askreddit", "2022-06")
        assert key in result.cells
        m = result.cells[key]
        assert m.total_comments == 10
        assert m.unique_authors == 2
        assert m.top1_share == pytest.approx(0.8)

    def test_skips_deleted_authors(self, tmp_path: Path):
        records = [
            {"id": "c1", "subreddit": "s", "author": "[deleted]", "created_utc": 1654100000},
            {"id": "c2", "subreddit": "s", "author": "[removed]", "created_utc": 1654100000},
            {"id": "c3", "subreddit": "s", "author": "real", "created_utc": 1654100000},
        ]
        p = tmp_path / "test.jsonl.zst"
        write_zst_jsonl(p, records)

        result = compute_helpers([p])
        m = result.cells[("s", "2022-06")]
        assert m.total_comments == 1
        assert m.unique_authors == 1

    def test_multiple_subreddits_months(self, tmp_path: Path):
        records = (
            _make_comments("AskReddit", {"u1": 3}, ts=1654100000)  # 2022-06
            + _make_comments("Depression", {"u2": 5}, ts=1656700000)  # 2022-07
        )
        p = tmp_path / "test.jsonl.zst"
        write_zst_jsonl(p, records)

        result = compute_helpers([p])
        assert len(result.cells) == 2
        subs = result.sorted_subreddits()
        assert "Depression" in subs
        assert "AskReddit" in subs


# ── analyse_helpers ──────────────────────────────────────────────────────────


class TestAnalyseHelpers:
    def _build_result(self) -> HelpersResult:
        """Build a small result with general + health subs over 4 months."""
        r = HelpersResult()

        # General: AskReddit — high concentration.
        for m_idx, month in enumerate(["2023-01", "2023-02", "2023-03", "2023-04"]):
            r.cells[("AskReddit", month)] = ConcentrationMetrics(
                total_comments=100 + m_idx * 10,
                unique_authors=20,
                top1_share=0.3,
                top5_share=0.6,
                hhi=0.15,
                gini=0.65,
                pct1_share=0.3,
                pct9_share=0.3,
                pct90_share=0.4,
            )

        # Health: Depression — lower concentration.
        for m_idx, month in enumerate(["2023-01", "2023-02", "2023-03", "2023-04"]):
            r.cells[("Depression", month)] = ConcentrationMetrics(
                total_comments=50 + m_idx * 5,
                unique_authors=30,
                top1_share=0.1,
                top5_share=0.25,
                hhi=0.05,
                gini=0.35,
                pct1_share=0.1,
                pct9_share=0.2,
                pct90_share=0.7,
            )

        return r

    def test_type_summaries(self):
        analysis = analyse_helpers(self._build_result())
        assert len(analysis.type_summaries) == 2
        types = {s.community_type for s in analysis.type_summaries}
        assert types == {"general", "health"}

        general = next(s for s in analysis.type_summaries if s.community_type == "general")
        assert general.n_subreddits == 1
        assert general.mean_gini == pytest.approx(0.65)

        health = next(s for s in analysis.type_summaries if s.community_type == "health")
        assert health.n_subreddits == 1
        assert health.mean_gini == pytest.approx(0.35)

    def test_moderation_rows(self):
        analysis = analyse_helpers(self._build_result())
        assert len(analysis.moderation_rows) == 2

        rows_by_sub = {r.subreddit: r for r in analysis.moderation_rows}
        assert "AskReddit" in rows_by_sub
        assert "Depression" in rows_by_sub

        ar = rows_by_sub["AskReddit"]
        assert ar.community_type == "general"
        assert ar.total_comments_first_half > 0
        assert ar.total_comments_second_half > 0

    def test_other_subreddits_excluded(self):
        r = HelpersResult()
        r.cells[("funny", "2023-01")] = ConcentrationMetrics(total_comments=100)
        r.cells[("funny", "2023-06")] = ConcentrationMetrics(total_comments=100)
        analysis = analyse_helpers(r)
        assert len(analysis.type_summaries) == 0
        assert len(analysis.moderation_rows) == 0


# ── View integration tests ───────────────────────────────────────────────────


class TestHelperViews:
    def _result(self) -> HelpersResult:
        r = HelpersResult()
        r.cells[("AskReddit", "2023-01")] = ConcentrationMetrics(
            total_comments=100, unique_authors=20,
            top1_share=0.3, top5_share=0.6, hhi=0.15, gini=0.65,
            pct1_share=0.3, pct9_share=0.3, pct90_share=0.4,
        )
        r.cells[("Depression", "2023-01")] = ConcentrationMetrics(
            total_comments=50, unique_authors=30,
            top1_share=0.1, top5_share=0.25, hhi=0.05, gini=0.35,
            pct1_share=0.1, pct9_share=0.2, pct90_share=0.7,
        )
        return r

    def test_write_monthly_csv(self, tmp_path: Path, monkeypatch):
        import views
        monkeypatch.setattr(views, "TABLES_DIR", tmp_path)

        from views import write_helpers_monthly_csv
        out = write_helpers_monthly_csv(self._result(), "test.csv")
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_write_type_summary_csv(self, tmp_path: Path, monkeypatch):
        import views
        monkeypatch.setattr(views, "TABLES_DIR", tmp_path)

        from views import write_helpers_type_summary_csv
        analysis = analyse_helpers(self._result())
        out = write_helpers_type_summary_csv(analysis, "test.csv")
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 types

    def test_write_moderation_csv(self, tmp_path: Path, monkeypatch):
        import views
        monkeypatch.setattr(views, "TABLES_DIR", tmp_path)

        from views import write_helpers_moderation_csv
        analysis = analyse_helpers(self._result())
        out = write_helpers_moderation_csv(analysis, "test.csv")
        assert out.exists()

    def test_plot_type_comparison(self, tmp_path: Path, monkeypatch):
        import views
        monkeypatch.setattr(views, "FIGURES_DIR", tmp_path)

        from views import plot_helpers_type_comparison
        analysis = analyse_helpers(self._result())
        out = plot_helpers_type_comparison(analysis, "test.svg")
        assert out.exists()

    def test_plot_moderation_scatter(self, tmp_path: Path, monkeypatch):
        import views
        monkeypatch.setattr(views, "FIGURES_DIR", tmp_path)

        from views import plot_helpers_moderation_scatter
        analysis = analyse_helpers(self._result())
        out = plot_helpers_moderation_scatter(analysis, "test.svg", metric="gini")
        assert out.exists()

    def test_plot_gini_trend(self, tmp_path: Path, monkeypatch):
        import views
        monkeypatch.setattr(views, "FIGURES_DIR", tmp_path)

        from views import plot_helpers_gini_trend
        out = plot_helpers_gini_trend(self._result(), "test.svg")
        assert out.exists()

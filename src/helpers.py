"""Repeat-helper concentration analysis for filtered Reddit data.

Streams filtered comment `.zst` files and counts comments per author per
(subreddit, month).  From the per-author distribution five concentration
metrics are derived:

1. **1-9-90 rule shares** — share of comments produced by the top 1 %
   ("power users"), the next 9 % ("occasional contributors"), and the
   remaining 90 % ("lurkers") of distinct authors.
2. **Top-1 share** — fraction of comments by the single most active author.
3. **Top-5 share** — fraction by the five most active authors.
4. **Herfindahl index** (HHI) — sum of squared author shares.
5. **Gini coefficient** — inequality of comment counts across authors.

Additionally, subreddits are classified into two community types:

- **general** — large general-discussion / entertainment communities
- **health**  — health-focused support and information communities

The module then relates concentration metrics to community type and to
activity change (comment volume moderation).
"""

from __future__ import annotations

import io
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import orjson
import zstandard as zstd

from src.config import (
    PROCESSED_DIR,
    ZSTD_MAX_WINDOW_SIZE,
)

log = logging.getLogger(__name__)


# ── Community-type classification ────────────────────────────────────────────

GENERAL_SUBREDDITS: frozenset[str] = frozenset({
    "AskReddit",
    "AmItheAsshole",
    "relationship_advice",
    "relationships",
    "legaladvice",
    "Advice",
    "teenagers",
    "childfree",
    "askscience",
    "science",
    "Fitness",
    "psychology",
})

HEALTH_SUBREDDITS: frozenset[str] = frozenset({
    "ADHD",
    "AlternativeHealth",
    "Amblyopia",
    "Anxiety",
    "AskDocs",
    "AskDoctorSmeeee",
    "Asthma",
    "BPD",
    "BabyBumps",
    "BipolarReddit",
    "CaregiverSupport",
    "ChronicPain",
    "CrohnsDisease",
    "Dentistry",
    "Depression",
    "DiagnoseMe",
    "Dystonia",
    "Fibromyalgia",
    "Gastroparesis",
    "Hemophilia",
    "Hypothyroidism",
    "Interstitialcystitis",
    "Keratoconus",
    "KidneyStones",
    "Menieres",
    "MultipleSclerosis",
    "Psoriasis",
    "STD",
    "SkincareAddiction",
    "SleepApnea",
    "SuicideWatch",
    "Testosterone",
    "aspergers",
    "birthcontrol",
    "cancer",
    "cfs",
    "chd",
    "dementia",
    "diabetes",
    "dysautonomia",
    "eczema",
    "flu",
    "gravesdisease",
    "healthcare",
    "ibs",
    "infertility",
    "lactoseintolerant",
    "maculardegeneration",
    "medical",
    "medicine",
    "mentalhealth",
    "nutrition",
    "optometry",
    "pancreaticcancer",
    "pharmacy",
    "pilonidalcyst",
    "pregnant",
    "publichealth",
    "rheumatoid",
    "stopsmoking",
    "transplant",
    "troubledteens",
})

_GENERAL_SUBREDDITS_NORMALIZED: frozenset[str] = frozenset(
    sub.casefold() for sub in GENERAL_SUBREDDITS
)
_HEALTH_SUBREDDITS_NORMALIZED: frozenset[str] = frozenset(
    sub.casefold() for sub in HEALTH_SUBREDDITS
)


def classify_subreddit(sub: str) -> str:
    """Return ``'general'``, ``'health'``, or ``'other'``."""
    normalized = sub.casefold()
    if normalized in _GENERAL_SUBREDDITS_NORMALIZED:
        return "general"
    if normalized in _HEALTH_SUBREDDITS_NORMALIZED:
        return "health"
    return "other"


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class ConcentrationMetrics:
    """Helper-concentration metrics for one (subreddit, month) cell."""

    total_comments: int = 0
    unique_authors: int = 0

    top1_share: float = 0.0
    top5_share: float = 0.0
    hhi: float = 0.0
    gini: float = 0.0

    # 1 / 9 / 90 percentile-based user-tier shares.
    pct1_share: float = 0.0   # share of comments by top 1 % of authors
    pct9_share: float = 0.0   # share by next 9 %
    pct90_share: float = 0.0  # share by remaining 90 %


@dataclass
class HelpersResult:
    """Full output of the repeat-helper analysis."""

    cells: dict[tuple[str, str], ConcentrationMetrics] = field(
        default_factory=dict,
    )

    def sorted_months(self) -> list[str]:
        """All months, sorted."""
        return sorted({m for _, m in self.cells})

    def sorted_subreddits(self) -> list[str]:
        """Subreddits ordered by total comments (descending)."""
        totals: Counter[str] = Counter()
        for (sub, _), c in self.cells.items():
            totals[sub] += c.total_comments
        return [s for s, _ in totals.most_common()]


# ── Metric computation ───────────────────────────────────────────────────────


def _compute_metrics(author_counts: Counter[str]) -> ConcentrationMetrics:
    """Derive all five concentration metrics from author comment counts."""
    m = ConcentrationMetrics()
    if not author_counts:
        return m

    total = sum(author_counts.values())
    m.total_comments = total
    m.unique_authors = len(author_counts)

    ranked = author_counts.most_common()  # descending by count

    # Top-1 / Top-5 shares.
    m.top1_share = ranked[0][1] / total if ranked else 0.0
    m.top5_share = sum(c for _, c in ranked[:5]) / total

    # Herfindahl index.
    m.hhi = sum((c / total) ** 2 for _, c in ranked)

    # Gini coefficient.
    m.gini = _gini([c for _, c in ranked])

    # 1 / 9 / 90 rule tiers.
    n = len(ranked)
    n1 = max(1, math.ceil(n * 0.01))
    n10 = max(n1 + 1, math.ceil(n * 0.10))

    top1_sum = sum(c for _, c in ranked[:n1])
    mid9_sum = sum(c for _, c in ranked[n1:n10])
    bot90_sum = sum(c for _, c in ranked[n10:])

    m.pct1_share = top1_sum / total
    m.pct9_share = mid9_sum / total
    m.pct90_share = bot90_sum / total

    return m


def _gini(values: list[int]) -> float:
    """Gini coefficient from a list of non-negative values."""
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_vals = sorted(values)
    cum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cum += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)


# ── Streaming engine ─────────────────────────────────────────────────────────


def _epoch_to_month(epoch: int) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def _extract_created_utc(record: dict) -> int | None:
    value = record.get("created_utc")
    if value is None:
        return None
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except Exception:
        return None


def _stream_zst(path: Path):  # noqa: ANN201
    """Yield parsed JSON objects from a .zst JSONL file."""
    dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
    with path.open("rb") as fin:
        with dctx.stream_reader(fin) as zin:
            buf = io.BufferedReader(zin)
            for line in buf:
                if not line.strip():
                    continue
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue


def _discover_filtered_paths(kind: str) -> list[Path]:
    """Find filtered .zst files for *kind* in ``PROCESSED_DIR``."""
    paths = sorted(PROCESSED_DIR.glob(f"filter-{kind}-*.jsonl.zst"))
    return [p for p in paths if not p.name.endswith(".progress.json")]


def compute_helpers(
    comment_paths: list[Path] | None = None,
) -> HelpersResult:
    """Stream filtered comments and compute per-author concentration metrics.

    Parameters
    ----------
    comment_paths
        Explicit .zst paths.  ``None`` auto-discovers from ``PROCESSED_DIR``.
    """
    if comment_paths is None:
        comment_paths = _discover_filtered_paths("comments")

    # Accumulate per-(sub, month) → Counter[author].
    author_counts: dict[tuple[str, str], Counter[str]] = {}

    for path in comment_paths:
        log.info("Streaming authors from %s …", path.name)
        for obj in _stream_zst(path):
            author = str(obj.get("author", ""))
            if not author or author in ("[deleted]", "[removed]"):
                continue
            sub: str = obj.get("subreddit", "unknown")
            ts = _extract_created_utc(obj)
            month = _epoch_to_month(ts) if ts is not None else "unknown"
            key = (sub, month)
            if key not in author_counts:
                author_counts[key] = Counter()
            author_counts[key][author] += 1

    result = HelpersResult()
    for key, counts in author_counts.items():
        result.cells[key] = _compute_metrics(counts)

    log.info(
        "Helpers: %d (subreddit, month) cells across %d subreddits",
        len(result.cells),
        len(result.sorted_subreddits()),
    )
    return result


# ── Moderation analysis ──────────────────────────────────────────────────────


@dataclass
class CommunityTypeSummary:
    """Aggregated concentration metrics for a community type."""

    community_type: str
    n_subreddits: int = 0
    mean_top1_share: float = 0.0
    mean_top5_share: float = 0.0
    mean_hhi: float = 0.0
    mean_gini: float = 0.0
    mean_pct1_share: float = 0.0
    mean_pct9_share: float = 0.0
    mean_pct90_share: float = 0.0


@dataclass
class ModerationRow:
    """One subreddit's concentration + activity change."""

    subreddit: str
    community_type: str
    mean_gini: float
    mean_top1_share: float
    mean_top5_share: float
    mean_hhi: float
    mean_pct1_share: float
    total_comments_first_half: int
    total_comments_second_half: int
    activity_change_pct: float


@dataclass
class HelpersAnalysis:
    """Combined output: type summaries + moderation rows."""

    type_summaries: list[CommunityTypeSummary] = field(default_factory=list)
    moderation_rows: list[ModerationRow] = field(default_factory=list)


def analyse_helpers(result: HelpersResult) -> HelpersAnalysis:
    """Aggregate helper metrics by community type and compute moderation."""
    analysis = HelpersAnalysis()
    months = result.sorted_months()

    # ── Per-type aggregation ─────────────────────────────────────────────
    type_accum: dict[str, list[ConcentrationMetrics]] = {}
    type_subs: dict[str, set[str]] = {}

    for (sub, _), cell in result.cells.items():
        ctype = classify_subreddit(sub)
        if ctype == "other":
            continue
        type_accum.setdefault(ctype, []).append(cell)
        type_subs.setdefault(ctype, set()).add(sub)

    for ctype in ("general", "health"):
        cells = type_accum.get(ctype, [])
        if not cells:
            continue
        n = len(cells)
        summary = CommunityTypeSummary(
            community_type=ctype,
            n_subreddits=len(type_subs.get(ctype, set())),
            mean_top1_share=sum(c.top1_share for c in cells) / n,
            mean_top5_share=sum(c.top5_share for c in cells) / n,
            mean_hhi=sum(c.hhi for c in cells) / n,
            mean_gini=sum(c.gini for c in cells) / n,
            mean_pct1_share=sum(c.pct1_share for c in cells) / n,
            mean_pct9_share=sum(c.pct9_share for c in cells) / n,
            mean_pct90_share=sum(c.pct90_share for c in cells) / n,
        )
        analysis.type_summaries.append(summary)

    # ── Moderation: concentration vs. activity change ────────────────────
    if not months:
        return analysis

    mid = len(months) // 2
    first_half = set(months[:mid])
    second_half = set(months[mid:])

    for sub in result.sorted_subreddits():
        ctype = classify_subreddit(sub)
        if ctype == "other":
            continue

        sub_cells = [
            c for (s, _), c in result.cells.items() if s == sub
        ]
        if not sub_cells:
            continue

        mean_gini = sum(c.gini for c in sub_cells) / len(sub_cells)
        mean_top1 = sum(c.top1_share for c in sub_cells) / len(sub_cells)
        mean_top5 = sum(c.top5_share for c in sub_cells) / len(sub_cells)
        mean_hhi = sum(c.hhi for c in sub_cells) / len(sub_cells)
        mean_pct1 = sum(c.pct1_share for c in sub_cells) / len(sub_cells)

        first_comments = sum(
            c.total_comments
            for (s, m), c in result.cells.items()
            if s == sub and m in first_half
        )
        second_comments = sum(
            c.total_comments
            for (s, m), c in result.cells.items()
            if s == sub and m in second_half
        )

        if first_comments == 0:
            continue

        change = (second_comments - first_comments) / first_comments * 100

        analysis.moderation_rows.append(
            ModerationRow(
                subreddit=sub,
                community_type=ctype,
                mean_gini=mean_gini,
                mean_top1_share=mean_top1,
                mean_top5_share=mean_top5,
                mean_hhi=mean_hhi,
                mean_pct1_share=mean_pct1,
                total_comments_first_half=first_comments,
                total_comments_second_half=second_comments,
                activity_change_pct=change,
            ),
        )

    # Stable sort.
    analysis.moderation_rows.sort(key=lambda r: r.subreddit)

    return analysis

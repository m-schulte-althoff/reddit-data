"""Create a stratified manual-coding sample for content proxy validation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import TABLES_DIR
from src.content_metrics import EXPERIENCE_RE, MEDICAL_RE, QUESTION_RE, SUPPORT_RE
from src.did import BASELINE_CUTOFF
from src.helpers import classify_subreddit
from src.io_utils import discover_filtered_paths, epoch_to_month, is_deleted_removed, stream_zst

DEFAULT_PER_STRATUM = 100
DEFAULT_SEED = 42


@dataclass
class ContentValidationArtifacts:
    """Manual-coding sample plus its stratum-level coverage table."""

    sample: pd.DataFrame
    coverage: pd.DataFrame
    table_paths: dict[str, Path]


def analyse_content_validation_sample(
    sample_path: Path | None = None,
    *,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """Summarize proxy precision and recall against completed manual labels."""
    out_tables = tables_dir or TABLES_DIR
    resolved_sample = sample_path or (out_tables / "content-validation-sample.csv")
    sample = pd.read_csv(resolved_sample, keep_default_na=False)
    rows: list[dict[str, int | float | str]] = []
    coverage_rows: list[dict[str, int | float | str]] = []
    for label in ("question", "experience", "support", "medical"):
        proxy_column = f"proxy_{label}"
        manual_column = f"manual_{label}"
        manual = pd.to_numeric(sample[manual_column], errors="coerce")
        invalid = manual.notna() & ~manual.isin([0, 1])
        if invalid.any():
            raise ValueError(f"{manual_column} must contain only 0, 1, or blanks")
        for (kind, community_type), group in sample.groupby(["kind", "community_type"], sort=True):
            coded_count = int(manual.loc[group.index].notna().sum())
            coverage_rows.append({
                "label": label,
                "kind": str(kind),
                "community_type": str(community_type),
                "n_sampled": int(len(group)),
                "n_coded": coded_count,
                "coding_rate": coded_count / len(group) if len(group) else 0.0,
            })
        coded = sample.loc[manual.notna(), ["kind", "community_type", proxy_column]].copy()
        coded[manual_column] = manual.loc[manual.notna()].astype(int)
        for (kind, community_type), group in coded.groupby(["kind", "community_type"], sort=True):
            true_positive = int(((group[proxy_column] == 1) & (group[manual_column] == 1)).sum())
            false_positive = int(((group[proxy_column] == 1) & (group[manual_column] == 0)).sum())
            false_negative = int(((group[proxy_column] == 0) & (group[manual_column] == 1)).sum())
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            rows.append({
                "label": label,
                "kind": str(kind),
                "community_type": str(community_type),
                "n_coded": int(len(group)),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": true_positive / precision_denominator if precision_denominator else float("nan"),
                "recall": true_positive / recall_denominator if recall_denominator else float("nan"),
            })
    result = pd.DataFrame(rows)
    result.to_csv(out_tables / "content-validation-report.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(
        out_tables / "content-validation-coding-coverage.csv",
        index=False,
    )
    return result


def create_content_validation_sample(
    comment_paths: list[Path] | None = None,
    submission_paths: list[Path] | None = None,
    *,
    per_stratum: int = DEFAULT_PER_STRATUM,
    seed: int = DEFAULT_SEED,
    tables_dir: Path | None = None,
) -> ContentValidationArtifacts:
    """Reservoir-sample health/general text by channel and pre/post period."""
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    resolved_comments = comment_paths or discover_filtered_paths("comments")
    resolved_submissions = submission_paths or discover_filtered_paths("submissions")
    reservoirs: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    seen: dict[tuple[str, str, str], int] = {}
    rng = random.Random(seed)

    _sample_paths(
        resolved_comments,
        kind="comment",
        per_stratum=per_stratum,
        reservoirs=reservoirs,
        seen=seen,
        rng=rng,
    )
    _sample_paths(
        resolved_submissions,
        kind="submission",
        per_stratum=per_stratum,
        reservoirs=reservoirs,
        seen=seen,
        rng=rng,
    )

    sample_rows = [row for rows in reservoirs.values() for row in rows]
    sample = pd.DataFrame(sample_rows).sort_values(
        ["kind", "community_type", "period", "subreddit", "record_id"],
        kind="stable",
    )
    coverage_rows = [
        {
            "kind": key[0],
            "community_type": key[1],
            "period": key[2],
            "eligible_records": count,
            "sampled_records": len(reservoirs.get(key, [])),
            "target_per_stratum": per_stratum,
        }
        for key, count in sorted(seen.items())
    ]
    coverage = pd.DataFrame(coverage_rows)
    out_tables = tables_dir or TABLES_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "sample": out_tables / "content-validation-sample.csv",
        "coverage": out_tables / "content-validation-coverage.csv",
    }
    sample.to_csv(table_paths["sample"], index=False)
    coverage.to_csv(table_paths["coverage"], index=False)
    return ContentValidationArtifacts(sample=sample, coverage=coverage, table_paths=table_paths)


def _sample_paths(
    paths: list[Path],
    *,
    kind: str,
    per_stratum: int,
    reservoirs: dict[tuple[str, str, str], list[dict[str, Any]]],
    seen: dict[tuple[str, str, str], int],
    rng: random.Random,
) -> None:
    """Add one channel to a deterministic per-stratum reservoir sample."""
    for path in paths:
        for record in stream_zst(path):
            row = _coding_row(record, kind=kind)
            if row is None:
                continue
            key = (kind, str(row["community_type"]), str(row["period"]))
            count = seen.get(key, 0)
            seen[key] = count + 1
            reservoir = reservoirs.setdefault(key, [])
            if len(reservoir) < per_stratum:
                reservoir.append(row)
                continue
            replacement = rng.randrange(count + 1)
            if replacement < per_stratum:
                reservoir[replacement] = row


def _coding_row(record: dict[str, Any], *, kind: str) -> dict[str, Any] | None:
    """Return a coding-ready row for one eligible record."""
    subreddit = str(record.get("subreddit", ""))
    community_type = classify_subreddit(subreddit)
    if community_type not in {"health", "general"}:
        return None
    if kind == "submission":
        text = f"{record.get('title', '')} {record.get('selftext', '')}".strip()
    else:
        text = str(record.get("body", ""))
    if not text or is_deleted_removed(text):
        return None
    created_utc = record.get("created_utc")
    if created_utc is None:
        return None
    try:
        month = epoch_to_month(int(float(created_utc)))
    except (TypeError, ValueError):
        return None
    return {
        "record_id": str(record.get("id", "")),
        "kind": kind,
        "subreddit": subreddit,
        "community_type": community_type,
        "month": month,
        "period": "post" if month >= BASELINE_CUTOFF else "pre",
        "text": text,
        "proxy_question": int(bool(QUESTION_RE.search(text))),
        "proxy_experience": int(bool(EXPERIENCE_RE.search(text))),
        "proxy_support": int(bool(SUPPORT_RE.search(text))),
        "proxy_medical": int(bool(MEDICAL_RE.search(text))),
        "manual_question": "",
        "manual_experience": "",
        "manual_support": "",
        "manual_medical": "",
        "coder_notes": "",
    }
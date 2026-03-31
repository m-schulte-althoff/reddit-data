"""Hugging Face extraction for Arctic reddit data."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import duckdb
from huggingface_hub import HfFileSystem, list_repo_files

from src.config import (
    END_EXCLUSIVE_TS,
    HF_REPO_ID,
    HF_REPO_TYPE,
    PROCESSED_DIR,
    START_TS,
)

log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_string_list(values: Iterable[str]) -> str:
    return ", ".join(_sql_quote(v) for v in values)


def _hf_uri(repo_path: str) -> str:
    return f"hf://datasets/{HF_REPO_ID}/{repo_path}"


def _iter_months(
    start_ts: "datetime",  # noqa: F821
    end_exclusive_ts: "datetime",  # noqa: F821
) -> list[tuple[int, int]]:
    """Return (year, month) tuples for every calendar month in the window."""
    from datetime import timedelta as _td

    end_incl = end_exclusive_ts - _td(microseconds=1)
    y, m = start_ts.year, start_ts.month
    months: list[tuple[int, int]] = []
    while (y, m) <= (end_incl.year, end_incl.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _month_prefix(year: int, month: int, kind: str) -> str:
    return f"data/{kind}/{year:04d}/{month:02d}/"


def _list_month_files(
    repo_files: list[str],
    months: list[tuple[int, int]],
    kind: str,
) -> dict[str, list[str]]:
    """Map YYYY-MM -> list of concrete hf:// parquet URIs."""
    parquet_files = [p for p in repo_files if p.endswith(".parquet")]
    result: dict[str, list[str]] = {}
    for y, m in months:
        prefix = _month_prefix(y, m, kind)
        ym = f"{y:04d}-{m:02d}"
        result[ym] = sorted(_hf_uri(p) for p in parquet_files if p.startswith(prefix))
    return result


def _ensure_complete(
    month_to_files: dict[str, list[str]],
    kind: str,
) -> None:
    missing = [mo for mo, files in month_to_files.items() if not files]
    if missing:
        log.warning(
            "Missing %s months in %s: %s — skipping.",
            kind,
            HF_REPO_ID,
            ", ".join(missing),
        )
        for mo in missing:
            del month_to_files[mo]


def _flatten(month_to_files: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for mo in sorted(month_to_files):
        out.extend(month_to_files[mo])
    return out


def _export_window(
    con: duckdb.DuckDBPyConnection,
    kind: str,
    uris: list[str],
    out_path: Path,
) -> dict:
    """Run a COPY … TO PARQUET filtered to the exact timestamp window."""
    if not uris:
        raise ValueError(f"No input files for {kind}.")
    uri_sql = _sql_string_list(uris)
    start_sql = START_TS.strftime("%Y-%m-%d %H:%M:%S")
    end_sql = END_EXCLUSIVE_TS.strftime("%Y-%m-%d %H:%M:%S")
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet([{uri_sql}], union_by_name=true)
            WHERE created_at >= TIMESTAMP {_sql_quote(start_sql)}
              AND created_at <  TIMESTAMP {_sql_quote(end_sql)}
        )
        TO {_sql_quote(out_path.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    stats = con.execute(f"""
        SELECT COUNT(*), MIN(created_at), MAX(created_at)
        FROM read_parquet({_sql_quote(out_path.as_posix())})
    """).fetchone()
    assert stats is not None
    return {
        "rows": int(stats[0]),
        "min_created_at": None if stats[1] is None else str(stats[1]),
        "max_created_at": None if stats[2] is None else str(stats[2]),
        "output_file": out_path.as_posix(),
    }


# ── Public API ───────────────────────────────────────────────────────────────

def list_available() -> None:
    """Print all available comment/submission months in the HF repo."""
    repo_files = list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE)
    parquet_files = sorted(p for p in repo_files if p.endswith(".parquet"))
    comments = [p for p in parquet_files if p.startswith("data/comments/")]
    submissions = [p for p in parquet_files if p.startswith("data/submissions/")]

    def _months(paths: list[str]) -> list[str]:
        return sorted({p.split("/")[2] + "-" + p.split("/")[3] for p in paths})

    log.info("Available comments months: %s (%d files)", _months(comments), len(comments))
    log.info("Available submissions months: %s (%d files)", _months(submissions), len(submissions))


def extract() -> None:
    """Download and filter parquet data from HuggingFace."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    start_fmt = START_TS.strftime("%Y%m%d")
    end_fmt = (END_EXCLUSIVE_TS - timedelta(seconds=1)).strftime("%Y%m%d")

    out_dir = PROCESSED_DIR
    comments_out = out_dir / f"hf_comments_{start_fmt}_{end_fmt}.parquet"
    submissions_out = out_dir / f"hf_submissions_{start_fmt}_{end_fmt}.parquet"
    manifest_out = out_dir / "hf_manifest.json"
    db_path = out_dir / "hf_extract.duckdb"

    months = _iter_months(START_TS, END_EXCLUSIVE_TS)
    repo_files = list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE)

    comment_mf = _list_month_files(repo_files, months, "comments")
    submission_mf = _list_month_files(repo_files, months, "submissions")
    _ensure_complete(comment_mf, "comments")
    _ensure_complete(submission_mf, "submissions")

    comment_uris = _flatten(comment_mf)
    submission_uris = _flatten(submission_mf)

    fs = HfFileSystem()
    duckdb.register_filesystem(fs)
    con = duckdb.connect(db_path.as_posix())
    for stmt in ("INSTALL httpfs;", "LOAD httpfs;"):
        try:
            con.execute(stmt)
        except Exception:
            pass

    log.info("Exporting comments …")
    c_stats = _export_window(con, "comments", comment_uris, comments_out)
    log.info("Exporting submissions …")
    s_stats = _export_window(con, "submissions", submission_uris, submissions_out)

    ym_labels = [f"{y:04d}-{m:02d}" for y, m in months]
    manifest = {
        "repo_id": HF_REPO_ID,
        "window": {
            "start_inclusive_utc": START_TS.isoformat(),
            "end_exclusive_utc": END_EXCLUSIVE_TS.isoformat(),
        },
        "months_touched": ym_labels,
        "outputs": {"comments": c_stats, "submissions": s_stats},
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("HF extraction complete. Wrote %s", manifest_out)

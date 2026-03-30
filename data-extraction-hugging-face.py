#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb",
#     "huggingface-hub",
# ]
# ///

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import duckdb
from huggingface_hub import HfFileSystem, list_repo_files


REPO_ID = "open-index/arctic"
REPO_TYPE = "dataset"

#START_TS = datetime(2021, 5, 30, 0, 0, 0, tzinfo=timezone.utc)
#END_EXCLUSIVE_TS = datetime(2023, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
START_TS = datetime(2022, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
END_EXCLUSIVE_TS = datetime(2022, 12, 30, 0, 0, 0, tzinfo=timezone.utc)

#OUTDIR = Path("arctic_exact_20210530_20231130")
OUTDIR = Path("arctic_exact_20221001_20221230")
COMMENTS_OUT = OUTDIR / "comments.parquet"
SUBMISSIONS_OUT = OUTDIR / "submissions.parquet"
MANIFEST_OUT = OUTDIR / "manifest.json"
DB_PATH = OUTDIR / "extract.duckdb"


@dataclass(frozen=True)
class MonthRef:
    year: int
    month: int

    @property
    def ym(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def prefix(self, kind: str) -> str:
        return f"data/{kind}/{self.year:04d}/{self.month:02d}/"


def iter_months(start_ts: datetime, end_exclusive_ts: datetime) -> list[MonthRef]:
    """
    Return all calendar months touched by the exact timestamp window.
    """
    end_inclusive = end_exclusive_ts - timedelta(microseconds=1)

    year = start_ts.year
    month = start_ts.month
    months: list[MonthRef] = []

    while (year, month) <= (end_inclusive.year, end_inclusive.month):
        months.append(MonthRef(year=year, month=month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return months


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_string_list(values: Iterable[str]) -> str:
    return ", ".join(sql_quote(v) for v in values)


def hf_uri(repo_path: str) -> str:
    return f"hf://datasets/{REPO_ID}/{repo_path}"


def list_month_files(repo_files: list[str], months: list[MonthRef], kind: str) -> dict[str, list[str]]:
    """
    Return a dict: YYYY-MM -> list of concrete hf:// parquet URIs for that month/type.
    """
    parquet_files = [p for p in repo_files if p.endswith(".parquet")]
    month_to_files: dict[str, list[str]] = {}

    for month_ref in months:
        prefix = month_ref.prefix(kind)
        files = sorted(hf_uri(p) for p in parquet_files if p.startswith(prefix))
        month_to_files[month_ref.ym] = files

    return month_to_files


def ensure_complete(month_to_files: dict[str, list[str]], kind: str) -> None:
    missing = [month for month, files in month_to_files.items() if not files]
    if missing:
        print(
            f"WARNING: Missing {kind} months in {REPO_ID}: {', '.join(missing)}. "
            f"Skipping those months."
        )
        for m in missing:
            del month_to_files[m]


def flatten_month_files(month_to_files: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for month in sorted(month_to_files.keys()):
        out.extend(month_to_files[month])
    return out


def export_exact_window(con: duckdb.DuckDBPyConnection, kind: str, uris: list[str], out_path: Path) -> dict:
    if not uris:
        raise ValueError(f"No input files provided for {kind}.")

    uri_sql = sql_string_list(uris)
    start_sql = START_TS.strftime("%Y-%m-%d %H:%M:%S")
    end_sql = END_EXCLUSIVE_TS.strftime("%Y-%m-%d %H:%M:%S")

    query = f"""
    COPY (
        SELECT *
        FROM read_parquet([{uri_sql}], union_by_name=true)
        WHERE created_at >= TIMESTAMP {sql_quote(start_sql)}
          AND created_at <  TIMESTAMP {sql_quote(end_sql)}
    )
    TO {sql_quote(out_path.as_posix())}
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)

    stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            MIN(created_at) AS min_created_at,
            MAX(created_at) AS max_created_at
        FROM read_parquet({sql_quote(out_path.as_posix())})
        """
    ).fetchone()

    return {
        "rows": int(stats[0]),
        "min_created_at": None if stats[1] is None else str(stats[1]),
        "max_created_at": None if stats[2] is None else str(stats[2]),
        "output_file": out_path.as_posix(),
    }


def list_available() -> None:
    """Print all available comment and submission months in the HF repo."""
    repo_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    parquet_files = sorted(p for p in repo_files if p.endswith(".parquet"))

    comments, submissions = [], []
    for p in parquet_files:
        if p.startswith("data/comments/"):
            comments.append(p)
        elif p.startswith("data/submissions/"):
            submissions.append(p)

    def months_from_paths(paths: list[str]) -> sorted:
        return sorted({p.split("/")[2] + "-" + p.split("/")[3] for p in paths})

    print("Available comments months:")
    for m in months_from_paths(comments):
        print(f"  {m}")
    print(f"  ({len(comments)} parquet files)\n")

    print("Available submissions months:")
    for m in months_from_paths(submissions):
        print(f"  {m}")
    print(f"  ({len(submissions)} parquet files)")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    months = iter_months(START_TS, END_EXCLUSIVE_TS)

    # One repo listing call, then strict local filtering.
    repo_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)

    comment_month_files = list_month_files(repo_files, months, "comments")
    submission_month_files = list_month_files(repo_files, months, "submissions")

    # Strict completeness check: fail if any month/type pair is absent.
    ensure_complete(comment_month_files, "comments")
    ensure_complete(submission_month_files, "submissions")

    comment_uris = flatten_month_files(comment_month_files)
    submission_uris = flatten_month_files(submission_month_files)

    # Register HF filesystem so DuckDB can read hf:// URIs.
    fs = HfFileSystem()
    duckdb.register_filesystem(fs)

    con = duckdb.connect(DB_PATH.as_posix())

    # Safe to try; if already installed/loaded, DuckDB handles that gracefully.
    try:
        con.execute("INSTALL httpfs;")
    except Exception:
        pass
    try:
        con.execute("LOAD httpfs;")
    except Exception:
        pass

    comment_stats = export_exact_window(con, "comments", comment_uris, COMMENTS_OUT)
    submission_stats = export_exact_window(con, "submissions", submission_uris, SUBMISSIONS_OUT)

    manifest = {
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "window": {
            "start_inclusive_utc": START_TS.isoformat(),
            "end_exclusive_utc": END_EXCLUSIVE_TS.isoformat(),
        },
        "months_touched": [m.ym for m in months],
        "inputs": {
            "comment_month_counts": {k: len(v) for k, v in comment_month_files.items()},
            "submission_month_counts": {k: len(v) for k, v in submission_month_files.items()},
            "comment_input_files_total": len(comment_uris),
            "submission_input_files_total": len(submission_uris),
        },
        "outputs": {
            "comments": comment_stats,
            "submissions": submission_stats,
        },
    }

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    import sys
    if "--list-available" in sys.argv:
        list_available()
    else:
        main()
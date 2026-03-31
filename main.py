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
    download    Download missing raw .zst files via Arctic Shift torrent.
    verify      Check that all raw files are present and valid.
    filter      Filter raw data to the configured time window.
    analyse     Compute descriptive statistics for comments and submissions.
    sample      Reservoir-sample records and write CSV + optional DataFrame.
    hf-extract  Extract data via Hugging Face (alternative source).
    hf-list     List months available on Hugging Face.
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


COMMANDS: dict[str, tuple[callable, str]] = {  # type: ignore[type-arg]
    "download": (cmd_download, "Download missing raw .zst files via torrent"),
    "verify": (cmd_verify, "Check raw files are present and valid"),
    "filter": (cmd_filter, "Filter raw data to configured time window"),
    "analyse": (cmd_analyse, "Descriptive statistics for raw data"),
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

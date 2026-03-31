"""Views — output formatting for tables and sample data."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from src.config import TABLES_DIR

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

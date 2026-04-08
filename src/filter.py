"""Filter raw Reddit .zst data by subreddit list and time window.

Streams compressed JSONL files line-by-line (never loads all data) and
writes matching records to ``data/processed/``.

Resumability
------------
A sidecar ``<output>.progress.json`` tracks which input files have been
fully processed.  On resume the completed files are skipped and new
frames are appended to the same output.  Multiple zstd frames in one
file are valid and decompress transparently.

Performance
-----------
- Byte-level regex extracts ``subreddit_name_prefixed`` *without* a full
  JSON parse; only matching lines are decoded with ``orjson``.
- A 16 MB ``io.BufferedReader`` reduces syscall overhead.
- Matching records are written in streaming fashion, keeping memory
  well below 1 GB even for 30 GB inputs.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import re
import time
from pathlib import Path

import zstandard as zstd

from src.config import (
    FILTER_READ_BUFFER,
    OUTPUT_ZSTD_LEVEL,
    PROCESSED_DIR,
    RAW_DIR,
    ZSTD_MAX_WINDOW_SIZE,
)

log = logging.getLogger(__name__)

# Byte-level regex to extract subreddit_name_prefixed without full JSON parse.
_SUB_PREFIX_RE = re.compile(rb'"subreddit_name_prefixed"\s*:\s*"([^"]+)"')

# Byte-level regex for created_utc (reused from analysis.py pattern).
_TS_RE = re.compile(rb'"created_utc"\s*:\s*"?(\d+)"?')


# ── Subreddit list I/O ──────────────────────────────────────────────────────


def load_subreddit_list(path: Path) -> set[str]:
    """Load subreddit names from a Python file defining ``subreddits = [...]``.

    Returns a **lowercase** set of prefixed names (e.g. ``{"r/askreddit", …}``).
    """
    text = path.read_text(encoding="utf-8")
    # Extract the list literal after 'subreddits = '
    match = re.search(r"subreddits\s*=\s*(\[.*?\])", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find 'subreddits = [...]' in {path}")
    raw_list: list[str] = ast.literal_eval(match.group(1))
    return {s.lower() for s in raw_list}


def _subreddit_set_to_bytes(subreddits: set[str]) -> set[bytes]:
    """Convert subreddit name set to bytes for matching against raw lines."""
    return {s.encode("utf-8") for s in subreddits}


# ── Path resolution ─────────────────────────────────────────────────────────


def _discover_raw_paths(
    kind: str,
    raw_dir: Path | None = None,
    months: list[tuple[int, int]] | None = None,
) -> list[Path]:
    """Find existing .zst files for *kind* (``"comments"`` or ``"submissions"``).

    If *months* is given, restrict to those months only.
    Otherwise return every .zst file present on disk.
    """
    base = (raw_dir or RAW_DIR) / "reddit" / kind
    prefix = "RC" if kind == "comments" else "RS"

    if months is not None:
        paths = []
        for y, m in sorted(months):
            p = base / f"{prefix}_{y:04d}-{m:02d}.zst"
            if p.exists():
                paths.append(p)
        return paths

    return sorted(base.glob(f"{prefix}_*.zst"))


# ── Progress tracking ───────────────────────────────────────────────────────


def _progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def _load_progress(output_path: Path) -> dict:
    pp = _progress_path(output_path)
    if pp.exists():
        return json.loads(pp.read_text(encoding="utf-8"))
    return {"completed_files": [], "rows_written": 0}


def _save_progress(output_path: Path, progress: dict) -> None:
    pp = _progress_path(output_path)
    pp.write_text(json.dumps(progress, indent=2), encoding="utf-8")


# ── Core filter engine ──────────────────────────────────────────────────────


def _filter_file(
    input_path: Path,
    zout: zstd.ZstdCompressionWriter,
    subreddits_bytes: set[bytes],
    start_epoch: int | None,
    end_epoch: int | None,
) -> dict:
    """Stream one .zst file, write matching lines to *zout*.

    Returns per-file stats dict.
    """
    rows_read = 0
    rows_written = 0
    t0 = time.monotonic()

    dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
    with input_path.open("rb") as fin:
        with dctx.stream_reader(fin) as zin:
            buf = io.BufferedReader(zin, buffer_size=FILTER_READ_BUFFER)
            for line in buf:
                if not line.strip():
                    continue
                rows_read += 1

                # ── Fast byte-level subreddit check ──
                m_sub = _SUB_PREFIX_RE.search(line)
                if m_sub is None:
                    continue
                sub_value = m_sub.group(1).lower()
                if sub_value not in subreddits_bytes:
                    continue

                # ── Optional time-window check (also byte-level) ──
                if start_epoch is not None or end_epoch is not None:
                    m_ts = _TS_RE.search(line)
                    if m_ts is None:
                        continue
                    ts = int(m_ts.group(1))
                    if start_epoch is not None and ts < start_epoch:
                        continue
                    if end_epoch is not None and ts >= end_epoch:
                        continue

                # Record matches — write the raw line (no re-serialisation).
                zout.write(line if line.endswith(b"\n") else line + b"\n")
                rows_written += 1

    elapsed = time.monotonic() - t0
    log.info(
        "  %s: %d matched / %d scanned (%.1fs)",
        input_path.name,
        rows_written,
        rows_read,
        elapsed,
    )
    return {
        "input_file": input_path.name,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "elapsed_seconds": round(elapsed, 1),
    }


# ── Public API ───────────────────────────────────────────────────────────────


def filter_by_subreddit(
    kind: str,
    subreddits: set[str],
    output_tag: str = "chan2025",
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    months: list[tuple[int, int]] | None = None,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    resume: bool = True,
) -> dict:
    """Filter raw .zst files for *kind* keeping only records in *subreddits*.

    Parameters
    ----------
    kind
        ``"comments"`` or ``"submissions"``.
    subreddits
        Lowercase prefixed names, e.g. ``{"r/askreddit", "r/science"}``.
    output_tag
        Short label used in the output filename.
    raw_dir
        Override base raw directory (for tests).
    output_dir
        Override processed output directory (for tests).
    months
        Restrict to specific months, e.g. ``[(2022, 6)]``.
        ``None`` means all available files on disk.
    start_epoch
        Optional inclusive start timestamp (UTC epoch seconds).
    end_epoch
        Optional exclusive end timestamp (UTC epoch seconds).
    resume
        When ``True``, skip already-completed files and append to output.

    Returns
    -------
    dict
        Summary with per-file stats, total rows read/written, output path.
    """
    out_dir = output_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"filter-{kind}-{output_tag}.jsonl.zst"
    output_path = out_dir / output_name

    all_paths = _discover_raw_paths(kind, raw_dir=raw_dir, months=months)
    if not all_paths:
        log.warning("No raw .zst files found for %s", kind)
        return {"output_file": str(output_path), "rows_read": 0, "rows_written": 0, "per_file": []}

    # ── Resume logic ──
    progress = _load_progress(output_path) if resume else {"completed_files": [], "rows_written": 0}
    completed_set = set(progress["completed_files"])
    pending = [p for p in all_paths if p.name not in completed_set]

    if not pending:
        log.info("All files already processed for %s — nothing to do.", output_name)
        return {
            "output_file": str(output_path),
            "rows_read": 0,
            "rows_written": progress["rows_written"],
            "per_file": [],
            "resumed": True,
        }

    log.info(
        "Filtering %s: %d file(s) pending (%d already done)",
        kind,
        len(pending),
        len(completed_set),
    )

    subreddits_bytes = _subreddit_set_to_bytes(subreddits)

    # Append mode ("ab") for resume; fresh write ("wb") otherwise.
    mode = "ab" if (resume and completed_set and output_path.exists()) else "wb"
    total_read = 0
    total_written = progress["rows_written"] if mode == "ab" else 0
    per_file: list[dict] = []

    cctx = zstd.ZstdCompressor(level=OUTPUT_ZSTD_LEVEL)
    with output_path.open(mode) as fout:
        for input_path in pending:
            # Each file gets its own zstd frame (valid multi-frame file).
            with cctx.stream_writer(fout, closefd=False) as zout:
                stats = _filter_file(
                    input_path,
                    zout,
                    subreddits_bytes,
                    start_epoch,
                    end_epoch,
                )
            per_file.append(stats)
            total_read += stats["rows_read"]
            total_written += stats["rows_written"]

            # Update progress after each file so a crash loses at most one.
            progress["completed_files"].append(input_path.name)
            progress["rows_written"] = total_written
            _save_progress(output_path, progress)

    log.info(
        "%s filter done: %d written / %d scanned → %s",
        kind,
        total_written,
        total_read,
        output_path,
    )
    return {
        "output_file": str(output_path),
        "rows_read": total_read,
        "rows_written": total_written,
        "per_file": per_file,
    }


def filter_all(
    output_tag: str = "chan2025",
    subreddit_list_path: Path | None = None,
    months: list[tuple[int, int]] | None = None,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    resume: bool = True,
) -> dict:
    """Filter both comments and submissions by the default subreddit list.

    Convenience wrapper around :func:`filter_by_subreddit`.
    """
    from src.config import SUBREDDIT_LIST_PATH

    list_path = subreddit_list_path or SUBREDDIT_LIST_PATH
    subreddits = load_subreddit_list(list_path)
    log.info("Loaded %d subreddits from %s", len(subreddits), list_path.name)

    results: dict[str, dict] = {}
    for kind in ("comments", "submissions"):
        results[kind] = filter_by_subreddit(
            kind=kind,
            subreddits=subreddits,
            output_tag=output_tag,
            months=months,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            resume=resume,
        )
    return results

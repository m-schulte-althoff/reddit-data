"""Arctic Shift torrent download, filtering, and verification."""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import libtorrent as lt
import orjson
import zstandard as zstd

from src.config import (
    DELETE_RAW_AFTER_FILTER,
    DOWNLOAD_POLL_SECONDS,
    END_EXCLUSIVE_EPOCH,
    END_EXCLUSIVE_TS,
    END_MONTH,
    LEGACY_BUNDLE_END_MONTH,
    METADATA_POLL_SECONDS,
    MONTHLY_TORRENT_INFOHASHES,
    OUTPUT_ZSTD_LEVEL,
    PROCESSED_DIR,
    RAW_DIR,
    START_EPOCH,
    START_MONTH,
    START_TS,
    TORRENT_INFOHASH,
    TRACKERS,
    ZSTD_MAX_WINDOW_SIZE,
)

log = logging.getLogger(__name__)
LEGACY_BUNDLE_NAME = "reddit-2005-06-to-2023-12"


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonthRef:
    year: int
    month: int

    @property
    def ym(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


@dataclass(frozen=True)
class TorrentBatch:
    infohash: str
    display_name: str
    target_paths: dict[str, list[str]]


@dataclass(frozen=True)
class ResolvedTorrentFile:
    index: int
    kind: str
    canonical_relpath: str
    actual_relpath: str


# ── Helpers ──────────────────────────────────────────────────────────────────

def iter_months(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[MonthRef]:
    """Yield every calendar month from *start* to *end* inclusive."""
    sy, sm = start
    ey, em = end
    months: list[MonthRef] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(MonthRef(y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def build_target_paths() -> dict[str, list[str]]:
    """Return torrent-relative paths for all target monthly files."""
    months = iter_months(START_MONTH, END_MONTH)
    return {
        "comments": [f"reddit/comments/RC_{m.ym}.zst" for m in months],
        "submissions": [f"reddit/submissions/RS_{m.ym}.zst" for m in months],
    }


def build_magnet_uri(
    infohash: str,
    trackers: Iterable[str],
    display_name: str = LEGACY_BUNDLE_NAME,
) -> str:
    """Construct a magnet URI from an info-hash and tracker list."""
    parts = [f"magnet:?xt=urn:btih:{infohash}", f"dn={display_name}"]
    parts.extend(f"tr={t}" for t in trackers)
    return "&".join(parts)


def _complete_marker(path: Path) -> Path:
    """Return the sidecar marker path for a downloaded file."""
    return path.with_suffix(path.suffix + ".complete")


def _is_complete(path: Path) -> bool:
    """Return True if *path* exists and has a .complete marker."""
    return path.exists() and _complete_marker(path).exists()


def missing_target_paths(
    target_paths: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Return target paths that are missing or incomplete on disk."""
    return {
        kind: [r for r in relpaths if not _is_complete(RAW_DIR / r)]
        for kind, relpaths in target_paths.items()
    }


def _human_bytes(num: int) -> str:
    n = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024.0 or unit == "PB":
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{num}B"


def _month_from_relpath(relpath: str) -> MonthRef:
    match = re.search(r"R[CS]_(\d{4})-(\d{2})\.zst$", relpath)
    if match is None:
        raise ValueError(f"Could not extract month from path: {relpath}")
    return MonthRef(int(match.group(1)), int(match.group(2)))


def _split_torrent_batches(target_paths: dict[str, list[str]]) -> list[TorrentBatch]:
    legacy_targets: dict[str, list[str]] = {"comments": [], "submissions": []}
    monthly_targets: dict[str, dict[str, list[str]]] = {}

    for kind, relpaths in target_paths.items():
        for relpath in relpaths:
            month = _month_from_relpath(relpath)
            if (month.year, month.month) <= LEGACY_BUNDLE_END_MONTH:
                legacy_targets[kind].append(relpath)
                continue

            infohash = MONTHLY_TORRENT_INFOHASHES.get(month.ym)
            if infohash is None:
                raise RuntimeError(
                    f"No monthly torrent infohash configured for Arctic Shift month {month.ym}"
                )
            batch_targets = monthly_targets.setdefault(
                month.ym,
                {"comments": [], "submissions": []},
            )
            batch_targets[kind].append(relpath)

    batches: list[TorrentBatch] = []
    if any(legacy_targets.values()):
        batches.append(TorrentBatch(TORRENT_INFOHASH, LEGACY_BUNDLE_NAME, legacy_targets))
    for month_key in sorted(monthly_targets):
        batches.append(
            TorrentBatch(
                MONTHLY_TORRENT_INFOHASHES[month_key],
                f"reddit-{month_key}",
                monthly_targets[month_key],
            )
        )
    return batches


# ── Torrent session ─────────────────────────────────────────────────────────

def _create_session() -> lt.session:
    ses = lt.session()
    try:
        ses.listen_on(6881, 6891)
    except Exception:
        pass
    try:
        ses.start_dht()
        ses.add_dht_router("router.bittorrent.com", 6881)
        ses.add_dht_router("router.utorrent.com", 6881)
        ses.add_dht_router("dht.transmissionbt.com", 6881)
    except Exception:
        pass
    return ses


def _wait_for_metadata(
    ses: lt.session,
    magnet_uri: str,
) -> lt.torrent_handle:
    params = {
        "save_path": str(RAW_DIR),
        "storage_mode": lt.storage_mode_t.storage_mode_sparse,
    }
    handle = lt.add_magnet_uri(ses, magnet_uri, params)
    log.info("Waiting for torrent metadata …")
    while not handle.has_metadata():
        s = handle.status()
        log.info(
            "  state=%s peers=%d down=%s/s progress=%.2f%%",
            getattr(s, "state", "unknown"),
            getattr(s, "num_peers", 0),
            _human_bytes(getattr(s, "download_rate", 0)),
            getattr(s, "progress", 0.0) * 100,
        )
        time.sleep(METADATA_POLL_SECONDS)
    log.info("Torrent metadata acquired.")
    return handle


def _map_files(
    handle: lt.torrent_handle,
) -> tuple[lt.torrent_info, dict[str, int]]:
    ti = handle.get_torrent_info()
    fs = ti.files()
    index_by_path: dict[str, int] = {}
    for i in range(fs.num_files()):
        index_by_path[fs.file_path(i)] = i
    return ti, index_by_path


def _prioritize(
    handle: lt.torrent_handle,
    ti: lt.torrent_info,
    index_by_path: dict[str, int],
    target_paths: dict[str, list[str]],
) -> tuple[dict[str, list[int]], list[ResolvedTorrentFile]]:
    fs = ti.files()
    priorities = [0] * fs.num_files()
    selected: dict[str, list[int]] = {"comments": [], "submissions": []}
    resolved: list[ResolvedTorrentFile] = []
    for kind, relpaths in target_paths.items():
        for relpath in relpaths:
            matched_relpath = relpath
            if relpath in index_by_path:
                idx = index_by_path[relpath]
            else:
                basename = Path(relpath).name
                basename_matches = [p for p in index_by_path if Path(p).name == basename]
                if len(basename_matches) == 1:
                    matched_relpath = basename_matches[0]
                    idx = index_by_path[matched_relpath]
                elif basename_matches:
                    raise RuntimeError(
                        f"Target file basename is ambiguous in torrent metadata: {relpath}"
                    )
                else:
                    raise RuntimeError(
                        f"Target file not found in torrent metadata: {relpath}"
                    )
            priorities[idx] = 4
            selected[kind].append(idx)
            resolved.append(ResolvedTorrentFile(idx, kind, relpath, matched_relpath))
    handle.prioritize_files(priorities)
    return selected, resolved


def _wait_for_files(
    handle: lt.torrent_handle,
    ti: lt.torrent_info,
    selected_indices: dict[str, list[int]],
) -> None:
    fs = ti.files()
    needed = sorted(
        set(selected_indices["comments"] + selected_indices["submissions"])
    )
    log.info("Downloading %d target files …", len(needed))
    while True:
        try:
            progress = handle.file_progress(flags=1)
        except TypeError:
            progress = handle.file_progress()
        incomplete = []
        downloaded_bytes = 0
        total_bytes = 0
        for idx in needed:
            size = fs.file_size(idx)
            done = progress[idx]
            total_bytes += size
            downloaded_bytes += min(done, size)
            if done < size:
                incomplete.append(idx)
        s = handle.status()
        pct = (100.0 * downloaded_bytes / total_bytes) if total_bytes else 100.0
        log.info(
            "  progress=%.2f%% down=%s/s up=%s/s peers=%d remaining=%d",
            pct,
            _human_bytes(getattr(s, "download_rate", 0)),
            _human_bytes(getattr(s, "upload_rate", 0)),
            getattr(s, "num_peers", 0),
            len(incomplete),
        )
        if not incomplete:
            log.info("All target files downloaded.")
            return
        time.sleep(DOWNLOAD_POLL_SECONDS)


def _finalize_downloaded_files(index_to_paths: dict[int, tuple[Path, Path]]) -> None:
    for actual_path, canonical_path in index_to_paths.values():
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        if actual_path != canonical_path:
            if actual_path.exists():
                actual_path.replace(canonical_path)
            elif not canonical_path.exists():
                raise FileNotFoundError(
                    f"Downloaded file missing after torrent completed: {actual_path}"
                )
        if not canonical_path.exists():
            raise FileNotFoundError(f"Expected downloaded file not found: {canonical_path}")
        _complete_marker(canonical_path).write_text(str(canonical_path.stat().st_size))
        log.info("  %s complete — marker written.", canonical_path.name)


# ── Public API ───────────────────────────────────────────────────────────────

def download() -> None:
    """Download missing monthly .zst files via BitTorrent."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_paths = build_target_paths()

    log.info("Target window: %s -> %s", START_TS.isoformat(), END_EXCLUSIVE_TS.isoformat())
    log.info(
        "Target monthly files: comments=%d submissions=%d",
        len(target_paths["comments"]),
        len(target_paths["submissions"]),
    )

    missing = missing_target_paths(target_paths)
    total_missing = sum(len(v) for v in missing.values())

    if total_missing == 0:
        log.info("All target files already on disk — nothing to download.")
        return

    log.info("%d file(s) missing — starting torrent.", total_missing)
    for batch in _split_torrent_batches(missing):
        batch_total = sum(len(v) for v in batch.target_paths.values())
        magnet_uri = build_magnet_uri(batch.infohash, TRACKERS, batch.display_name)
        log.info(
            "Starting torrent batch %s for %d file(s).",
            batch.display_name,
            batch_total,
        )
        ses = _create_session()
        handle = _wait_for_metadata(ses, magnet_uri)
        ti, index_by_path = _map_files(handle)
        selected, resolved = _prioritize(handle, ti, index_by_path, batch.target_paths)

        idx_to_paths = {
            item.index: (
                RAW_DIR / item.actual_relpath,
                RAW_DIR / item.canonical_relpath,
            )
            for item in resolved
        }

        _wait_for_files(handle, ti, selected)
        _finalize_downloaded_files(idx_to_paths)

        try:
            ses.remove_torrent(handle)
        except Exception:
            pass

    log.info("Download complete.")


def _extract_created_utc(record: dict) -> int | None:
    value = record.get("created_utc")
    if value is None:
        return None
    try:
        return int(float(value)) if isinstance(value, str) else int(value)
    except Exception:
        return None


def _stream_filter(
    input_paths: list[Path],
    output_path: Path,
) -> dict:
    """Filter JSONL-zst files by timestamp window and write merged output."""
    rows_in = 0
    rows_out = 0
    min_ts: int | None = None
    max_ts: int | None = None
    per_file: list[dict] = []

    cctx = zstd.ZstdCompressor(level=OUTPUT_ZSTD_LEVEL)
    with output_path.open("wb") as fout:
        with cctx.stream_writer(fout) as zout:
            for input_path in input_paths:
                f_in = f_out = 0
                f_min: int | None = None
                f_max: int | None = None
                log.info("Filtering %s …", input_path.name)
                dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
                with input_path.open("rb") as fin:
                    with dctx.stream_reader(fin) as zin:
                        buf = io.BufferedReader(zin)
                        for line in buf:
                            if not line.strip():
                                continue
                            rows_in += 1
                            f_in += 1
                            try:
                                obj = orjson.loads(line)
                            except orjson.JSONDecodeError:
                                continue
                            ts = _extract_created_utc(obj)
                            if ts is None:
                                continue
                            if START_EPOCH <= ts < END_EXCLUSIVE_EPOCH:
                                zout.write(line)
                                rows_out += 1
                                f_out += 1
                                if min_ts is None or ts < min_ts:
                                    min_ts = ts
                                if max_ts is None or ts > max_ts:
                                    max_ts = ts
                                if f_min is None or ts < f_min:
                                    f_min = ts
                                if f_max is None or ts > f_max:
                                    f_max = ts
                per_file.append({
                    "input_file": str(input_path),
                    "rows_read": f_in,
                    "rows_written": f_out,
                    "min_created_utc": f_min,
                    "max_created_utc": f_max,
                })

    return {
        "output_file": str(output_path),
        "rows_read": rows_in,
        "rows_written": rows_out,
        "min_created_utc": min_ts,
        "max_created_utc": max_ts,
        "per_file": per_file,
    }


def filter_raw() -> None:
    """Filter downloaded raw .zst files into the configured time window."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target_paths = build_target_paths()

    start_fmt = START_TS.strftime("%Y%m%d")
    end_fmt = (END_EXCLUSIVE_TS - timedelta(seconds=1)).strftime("%Y%m%d")
    comments_out = PROCESSED_DIR / f"comments_{start_fmt}_{end_fmt}.jsonl.zst"
    submissions_out = PROCESSED_DIR / f"submissions_{start_fmt}_{end_fmt}.jsonl.zst"
    manifest_out = PROCESSED_DIR / "manifest.json"

    comment_paths = [RAW_DIR / r for r in target_paths["comments"]]
    submission_paths = [RAW_DIR / r for r in target_paths["submissions"]]

    for p in comment_paths + submission_paths:
        if not p.exists():
            raise FileNotFoundError(f"Raw file missing: {p}  — run 'download' first.")

    log.info("Filtering comments …")
    c_stats = _stream_filter(comment_paths, comments_out)
    log.info("Filtering submissions …")
    s_stats = _stream_filter(submission_paths, submissions_out)

    if DELETE_RAW_AFTER_FILTER:
        for p in comment_paths + submission_paths:
            p.unlink(missing_ok=True)

    manifest = {
        "source": {
            "name": "Arctic Shift torrent",
            "batches": [
                {
                    "display_name": batch.display_name,
                    "infohash": batch.infohash,
                    "magnet_uri": build_magnet_uri(
                        batch.infohash,
                        TRACKERS,
                        batch.display_name,
                    ),
                }
                for batch in _split_torrent_batches(target_paths)
            ],
        },
        "window": {
            "start_inclusive_utc": START_TS.isoformat(),
            "end_exclusive_utc": END_EXCLUSIVE_TS.isoformat(),
        },
        "selected_files": target_paths,
        "outputs": {"comments": c_stats, "submissions": s_stats},
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Filter complete. Wrote %s", manifest_out)


def verify() -> bool:
    """Check every target file: exists, non-empty, valid zstd header."""
    target_paths = build_target_paths()
    all_ok = True
    for kind, relpaths in target_paths.items():
        for rel in relpaths:
            path = RAW_DIR / rel
            label = f"[{kind}] {rel}"
            if not path.exists():
                log.warning("%s  MISSING", label)
                all_ok = False
                continue
            size = path.stat().st_size
            if size == 0:
                log.warning("%s  EMPTY", label)
                all_ok = False
                continue
            try:
                dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
                with path.open("rb") as f:
                    with dctx.stream_reader(f) as zin:
                        chunk = zin.read(4096)
                        if not chunk:
                            log.warning("%s  decompresses to 0 bytes", label)
                            all_ok = False
                            continue
            except Exception as exc:
                log.warning("%s  CORRUPT — %s", label, exc)
                all_ok = False
                continue
            log.info("%s  OK  %s", label, _human_bytes(size))
    return all_ok

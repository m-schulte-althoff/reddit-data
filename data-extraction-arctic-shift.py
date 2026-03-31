#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "libtorrent",
#     "orjson",
#     "zstandard",
# ]
# ///

from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import libtorrent as lt
import orjson
import zstandard as zstd


# =========================
# Configuration
# =========================

# Arctic Shift 2005-06 .. 2023-12 bundle torrent
TORRENT_INFOHASH = "9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4"

# Open trackers to make metadata/peer discovery more reliable.
TRACKERS = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://tracker.openbittorrent.com:6969/announce",
    "udp://tracker.torrent.eu.org:451/announce",
    "udp://open.stealth.si:80/announce",
]

# Exact requested window:
# 18 months before 2022-11-30 and 12 months after, interpreted literally.
#START_TS = datetime(2021, 5, 30, 0, 0, 0, tzinfo=timezone.utc)
#END_EXCLUSIVE_TS = datetime(2023, 12, 1, 0, 0, 0, tzinfo=timezone.utc)

START_TS = datetime(2022, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
END_EXCLUSIVE_TS = datetime(2023, 7, 30, 0, 0, 0, tzinfo=timezone.utc)

START_EPOCH = int(START_TS.timestamp())
END_EXCLUSIVE_EPOCH = int(END_EXCLUSIVE_TS.timestamp())

# Derive the month range from the timestamp window.
# START_TS  -> first month to download.
# END_EXCLUSIVE_TS is exclusive, so the last needed month contains (END_EXCLUSIVE_TS - 1 second).
START_MONTH = (START_TS.year, START_TS.month)
_end_incl = END_EXCLUSIVE_TS - timedelta(seconds=1)
END_MONTH = (_end_incl.year, _end_incl.month)

BASE_DIR = Path.cwd()
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

COMMENTS_OUT = OUT_DIR / "comments_20221001_20221230.jsonl.zst"
SUBMISSIONS_OUT = OUT_DIR / "submissions_20221001_20221230.jsonl.zst"
MANIFEST_OUT = OUT_DIR / "manifest.json"

# Delete the raw monthly .zst files after they have been filtered.
DELETE_RAW_AFTER_FILTER = False

# Poll intervals
METADATA_POLL_SECONDS = 3
DOWNLOAD_POLL_SECONDS = 10

# zstd output compression level
ZSTD_DECOMPRESSOR_MAX_WINDOW_SIZE = 2**31
OUTPUT_ZSTD_LEVEL = 9


# =========================
# Utilities
# =========================

@dataclass(frozen=True)
class MonthRef:
    year: int
    month: int

    @property
    def ym(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def iter_months(start: Tuple[int, int], end: Tuple[int, int]) -> List[MonthRef]:
    sy, sm = start
    ey, em = end
    months: List[MonthRef] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(MonthRef(y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def build_target_paths() -> Dict[str, List[str]]:
    months = iter_months(START_MONTH, END_MONTH)
    comments = [f"reddit/comments/RC_{m.ym}.zst" for m in months]
    submissions = [f"reddit/submissions/RS_{m.ym}.zst" for m in months]
    return {"comments": comments, "submissions": submissions}


def build_magnet_uri(infohash: str, trackers: Iterable[str]) -> str:
    parts = [f"magnet:?xt=urn:btih:{infohash}", "dn=reddit-2005-06-to-2023-12"]
    parts.extend(f"tr={tracker}" for tracker in trackers)
    return "&".join(parts)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def missing_target_paths(target_paths: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Return a dict with only the target paths not yet present on disk."""
    missing: Dict[str, List[str]] = {}
    for kind, relpaths in target_paths.items():
        missing[kind] = [rel for rel in relpaths if not (RAW_DIR / rel).exists()]
    return missing


def human_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    n = float(num)
    for unit in units:
        if n < 1024.0 or unit == units[-1]:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{num}B"


# =========================
# Torrent download
# =========================

def create_session() -> lt.session:
    ses = lt.session()
    try:
        ses.listen_on(6881, 6891)
    except Exception:
        # Some libtorrent builds do not expose listen_on the same way; defaults are usually fine.
        pass

    try:
        ses.start_dht()
        ses.add_dht_router("router.bittorrent.com", 6881)
        ses.add_dht_router("router.utorrent.com", 6881)
        ses.add_dht_router("dht.transmissionbt.com", 6881)
    except Exception:
        pass

    return ses


def add_torrent_and_wait_for_metadata(ses: lt.session, magnet_uri: str) -> lt.torrent_handle:
    params = {
        "save_path": str(RAW_DIR),
        "storage_mode": lt.storage_mode_t.storage_mode_sparse,
    }
    handle = lt.add_magnet_uri(ses, magnet_uri, params)

    print("Waiting for torrent metadata...")
    while not handle.has_metadata():
        s = handle.status()
        print(
            f"  state={getattr(s, 'state', 'unknown')} "
            f"peers={getattr(s, 'num_peers', 0)} "
            f"down={human_bytes(getattr(s, 'download_rate', 0))}/s "
            f"progress={getattr(s, 'progress', 0.0) * 100:.2f}%"
        )
        time.sleep(METADATA_POLL_SECONDS)

    print("Torrent metadata acquired.")
    return handle


def map_files(handle: lt.torrent_handle) -> Tuple[lt.torrent_info, Dict[str, int]]:
    ti = handle.get_torrent_info()
    fs = ti.files()
    index_by_path: Dict[str, int] = {}
    for i in range(fs.num_files()):
        index_by_path[fs.file_path(i)] = i
    return ti, index_by_path


def prioritize_only_targets(
    handle: lt.torrent_handle,
    ti: lt.torrent_info,
    index_by_path: Dict[str, int],
    target_paths: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    fs = ti.files()
    priorities = [0] * fs.num_files()
    selected: Dict[str, List[int]] = {"comments": [], "submissions": []}

    for kind, relpaths in target_paths.items():
        for relpath in relpaths:
            if relpath not in index_by_path:
                raise RuntimeError(f"Target file not found in torrent metadata: {relpath}")
            idx = index_by_path[relpath]
            priorities[idx] = 4
            selected[kind].append(idx)

    handle.prioritize_files(priorities)
    return selected


def wait_for_selected_files(
    handle: lt.torrent_handle,
    ti: lt.torrent_info,
    selected_indices: Dict[str, List[int]],
) -> None:
    fs = ti.files()
    needed = sorted(set(selected_indices["comments"] + selected_indices["submissions"]))

    print(f"Downloading {len(needed)} target files...")
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
        print(
            f"  selected_progress={pct:.2f}% "
            f"down={human_bytes(getattr(s, 'download_rate', 0))}/s "
            f"up={human_bytes(getattr(s, 'upload_rate', 0))}/s "
            f"peers={getattr(s, 'num_peers', 0)} "
            f"remaining_files={len(incomplete)}"
        )

        if not incomplete:
            print("All target files downloaded.")
            return

        time.sleep(DOWNLOAD_POLL_SECONDS)


# =========================
# Filtering
# =========================

def extract_created_utc(record: dict) -> int | None:
    value = record.get("created_utc")
    if value is None:
        return None
    try:
        if isinstance(value, str):
            # Some dumps encode as string; allow floats stored as strings as well.
            return int(float(value))
        return int(value)
    except Exception:
        return None


def stream_filter_jsonl_zst(
    input_paths: List[Path],
    output_path: Path,
) -> dict:
    rows_in = 0
    rows_out = 0
    min_created = None
    max_created = None
    per_file_stats = []

    cctx = zstd.ZstdCompressor(level=OUTPUT_ZSTD_LEVEL)

    with output_path.open("wb") as fout_raw:
        with cctx.stream_writer(fout_raw) as zout:
            for input_path in input_paths:
                file_in = 0
                file_out = 0
                file_min = None
                file_max = None

                print(f"Filtering {input_path} ...")
                with input_path.open("rb") as fin_raw:
                    dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_DECOMPRESSOR_MAX_WINDOW_SIZE)
                    with dctx.stream_reader(fin_raw) as zin:
                        buffered = io.BufferedReader(zin)

                        while True:
                            line = buffered.readline()
                            if not line:
                                break
                            if not line.strip():
                                continue

                            rows_in += 1
                            file_in += 1

                            try:
                                obj = orjson.loads(line)
                            except orjson.JSONDecodeError:
                                continue

                            created_utc = extract_created_utc(obj)
                            if created_utc is None:
                                continue

                            if START_EPOCH <= created_utc < END_EXCLUSIVE_EPOCH:
                                zout.write(line)
                                rows_out += 1
                                file_out += 1

                                if min_created is None or created_utc < min_created:
                                    min_created = created_utc
                                if max_created is None or created_utc > max_created:
                                    max_created = created_utc

                                if file_min is None or created_utc < file_min:
                                    file_min = created_utc
                                if file_max is None or created_utc > file_max:
                                    file_max = created_utc

                per_file_stats.append(
                    {
                        "input_file": str(input_path),
                        "rows_read": file_in,
                        "rows_written": file_out,
                        "min_created_utc_written": file_min,
                        "max_created_utc_written": file_max,
                    }
                )

    return {
        "output_file": str(output_path),
        "rows_read": rows_in,
        "rows_written": rows_out,
        "min_created_utc_written": min_created,
        "max_created_utc_written": max_created,
        "start_inclusive_utc": START_TS.isoformat(),
        "end_exclusive_utc": END_EXCLUSIVE_TS.isoformat(),
        "per_file_stats": per_file_stats,
    }


# =========================
# Verification
# =========================

def verify_downloads(target_paths: Dict[str, List[str]]) -> bool:
    """Check that every target file exists, is non-empty, and is valid zstd."""
    all_ok = True
    for kind, relpaths in target_paths.items():
        for rel in relpaths:
            path = RAW_DIR / rel
            label = f"  [{kind}] {rel}"

            if not path.exists():
                print(f"{label}  MISSING")
                all_ok = False
                continue

            size = path.stat().st_size
            if size == 0:
                print(f"{label}  EMPTY (0 bytes)")
                all_ok = False
                continue

            # Try to read the first few decompressed bytes to confirm a valid zstd frame.
            try:
                dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_DECOMPRESSOR_MAX_WINDOW_SIZE)
                with path.open("rb") as f:
                    with dctx.stream_reader(f) as zin:
                        chunk = zin.read(4096)
                        if not chunk:
                            print(f"{label}  WARN  {human_bytes(size)} on disk but decompresses to 0 bytes")
                            all_ok = False
                            continue
            except Exception as exc:
                print(f"{label}  CORRUPT  {human_bytes(size)} on disk — {exc}")
                all_ok = False
                continue

            print(f"{label}  OK  {human_bytes(size)}")

    return all_ok


# =========================
# Main
# =========================

def cmd_download() -> int:
    """Download missing monthly files via torrent."""
    ensure_dirs()
    target_paths = build_target_paths()
    magnet_uri = build_magnet_uri(TORRENT_INFOHASH, TRACKERS)

    print("Target window:")
    print(f"  start_inclusive_utc = {START_TS.isoformat()}")
    print(f"  end_exclusive_utc   = {END_EXCLUSIVE_TS.isoformat()}")

    print("\nTarget monthly files:")
    print(f"  comments   = {len(target_paths['comments'])}")
    print(f"  submissions= {len(target_paths['submissions'])}")

    missing = missing_target_paths(target_paths)
    total_missing = sum(len(v) for v in missing.values())

    if total_missing == 0:
        print("\nAll target files already present on disk — skipping torrent download.")
    else:
        print(f"\n{total_missing} file(s) missing — starting torrent download.")
        ses = create_session()
        handle = add_torrent_and_wait_for_metadata(ses, magnet_uri)
        ti, index_by_path = map_files(handle)
        selected = prioritize_only_targets(handle, ti, index_by_path, missing)
        wait_for_selected_files(handle, ti, selected)

        try:
            ses.remove_torrent(handle)
        except Exception:
            pass

    comment_input_paths = [RAW_DIR / rel for rel in target_paths["comments"]]
    submission_input_paths = [RAW_DIR / rel for rel in target_paths["submissions"]]

    for p in comment_input_paths + submission_input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Expected downloaded file missing: {p}")

    print("\nDownload complete.")
    return 0


def cmd_filter() -> int:
    """Filter downloaded raw files into the target time window and write compressed output."""
    ensure_dirs()
    target_paths = build_target_paths()
    magnet_uri = build_magnet_uri(TORRENT_INFOHASH, TRACKERS)

    comment_input_paths = [RAW_DIR / rel for rel in target_paths["comments"]]
    submission_input_paths = [RAW_DIR / rel for rel in target_paths["submissions"]]

    for p in comment_input_paths + submission_input_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Raw file missing: {p}\nRun 'download' first."
            )

    print("Filtering comments ...")
    comments_stats = stream_filter_jsonl_zst(comment_input_paths, COMMENTS_OUT)
    print("Filtering submissions ...")
    submissions_stats = stream_filter_jsonl_zst(submission_input_paths, SUBMISSIONS_OUT)

    if DELETE_RAW_AFTER_FILTER:
        for p in comment_input_paths + submission_input_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    manifest = {
        "source": {
            "name": "Arctic Shift 2005-06 to 2023-12 bundle torrent",
            "infohash": TORRENT_INFOHASH,
            "magnet_uri": magnet_uri,
        },
        "window": {
            "start_inclusive_utc": START_TS.isoformat(),
            "end_exclusive_utc": END_EXCLUSIVE_TS.isoformat(),
            "start_inclusive_epoch": START_EPOCH,
            "end_exclusive_epoch": END_EXCLUSIVE_EPOCH,
        },
        "selected_files": target_paths,
        "outputs": {
            "comments": comments_stats,
            "submissions": submissions_stats,
        },
    }

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  {COMMENTS_OUT}")
    print(f"  {SUBMISSIONS_OUT}")
    print(f"  {MANIFEST_OUT}")
    return 0


def cmd_verify() -> int:
    """Verify that all target files are present and valid zstd archives."""
    target_paths = build_target_paths()
    print("Verifying downloaded files ...\n")
    ok = verify_downloads(target_paths)
    if ok:
        print("\nAll files OK.")
        return 0
    else:
        print("\nSome files are missing or corrupt.")
        return 1


COMMANDS = {
    "download": cmd_download,
    "filter": cmd_filter,
    "verify": cmd_verify,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} <command>")
        print(f"Commands: {', '.join(COMMANDS)}")
        return 2
    return COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
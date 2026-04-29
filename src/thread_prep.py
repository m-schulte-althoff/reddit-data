"""Optional submission-hash partition prep for thread-oriented analyses.

The safe path for large filtered exports is to preprocess comments and
submissions into shard files keyed by the submission id. This keeps all records
for one thread together so downstream stages can process one shard at a time
without needing a global in-memory submission or comment graph.
"""

from __future__ import annotations

import json
import logging
import shutil
import zlib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import zstandard as zstd

from src.config import OUTPUT_DIR, OUTPUT_ZSTD_LEVEL
from src.io_utils import fingerprint_hash, fingerprint_paths, stream_zst

log = logging.getLogger(__name__)
_ORJSON_DUMPS = getattr(orjson, "dumps")

THREAD_PREP_CACHE_VERSION = 1
_MANIFEST_FILENAME = "thread-prep-manifest.json"


@dataclass(frozen=True)
class ThreadPrepConfig:
    """Configuration for optional submission-hash partition prep."""

    partitions: int
    cache_dir: Path | None = None

    @property
    def enabled(self) -> bool:
        """Return whether submission-hash partition prep should be used."""
        return self.partitions > 1


@dataclass(frozen=True)
class ThreadPrepArtifacts:
    """Prepared shard files plus cache metadata."""

    config: ThreadPrepConfig
    root_dir: Path
    manifest_path: Path
    metadata: dict[str, Any]
    comment_partitions: list[Path]
    submission_partitions: list[Path]


def normalize_thread_prep_config(
    partitions: int | None,
    *,
    cache_dir: Path | None = None,
) -> ThreadPrepConfig | None:
    """Return a normalized config or ``None`` when partition prep is disabled."""
    if partitions is None or partitions <= 1:
        return None
    if partitions < 0:
        raise ValueError("partitions must be >= 0")
    return ThreadPrepConfig(partitions=partitions, cache_dir=cache_dir)


def prepare_thread_partitions(
    comment_paths: list[Path],
    submission_paths: list[Path],
    *,
    config: ThreadPrepConfig,
) -> ThreadPrepArtifacts:
    """Create or reuse submission-hash shard files for comments and submissions."""
    if not config.enabled:
        raise ValueError("partition prep requires partitions > 1")

    root_dir = _root_dir_for_config(config)
    manifest_path = root_dir / _MANIFEST_FILENAME
    metadata = _metadata_payload(comment_paths, submission_paths, config)
    comment_partitions = _partition_paths(root_dir, "comments", config.partitions)
    submission_partitions = _partition_paths(root_dir, "submissions", config.partitions)

    if _has_valid_cache(
        manifest_path=manifest_path,
        metadata=metadata,
        comment_partitions=comment_partitions,
        submission_partitions=submission_partitions,
    ):
        log.info(
            "Using cached thread prep with %d submission-id partitions at %s",
            config.partitions,
            root_dir,
        )
        cached_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ThreadPrepArtifacts(
            config=config,
            root_dir=root_dir,
            manifest_path=manifest_path,
            metadata=cached_metadata,
            comment_partitions=comment_partitions,
            submission_partitions=submission_partitions,
        )

    if root_dir.exists():
        shutil.rmtree(root_dir)
    (root_dir / "comments").mkdir(parents=True, exist_ok=True)
    (root_dir / "submissions").mkdir(parents=True, exist_ok=True)

    _write_partition_files(comment_paths, comment_partitions, kind="comments")
    _write_partition_files(submission_paths, submission_partitions, kind="submissions")

    manifest_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Prepared thread prep with %d submission-id partitions at %s",
        config.partitions,
        root_dir,
    )
    return ThreadPrepArtifacts(
        config=config,
        root_dir=root_dir,
        manifest_path=manifest_path,
        metadata=metadata,
        comment_partitions=comment_partitions,
        submission_partitions=submission_partitions,
    )


def _root_dir_for_config(config: ThreadPrepConfig) -> Path:
    base_dir = config.cache_dir or (OUTPUT_DIR / "cache")
    return base_dir / "thread-prep" / f"submission-hash-{config.partitions:04d}"


def _partition_paths(root_dir: Path, kind: str, partitions: int) -> list[Path]:
    return [
        root_dir / kind / f"{kind}-{index:04d}-of-{partitions:04d}.jsonl.zst"
        for index in range(partitions)
    ]


def _metadata_payload(
    comment_paths: list[Path],
    submission_paths: list[Path],
    config: ThreadPrepConfig,
) -> dict[str, Any]:
    sources = {
        "version": THREAD_PREP_CACHE_VERSION,
        "partitions": config.partitions,
        "comment_files": fingerprint_paths(comment_paths),
        "submission_files": fingerprint_paths(submission_paths),
    }
    return {
        "sources": sources,
        "fingerprint": fingerprint_hash(sources),
    }


def _has_valid_cache(
    *,
    manifest_path: Path,
    metadata: dict[str, Any],
    comment_partitions: list[Path],
    submission_partitions: list[Path],
) -> bool:
    if not manifest_path.exists():
        return False
    if not all(path.exists() for path in comment_partitions + submission_partitions):
        return False

    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return existing.get("fingerprint") == metadata.get("fingerprint")


def _write_partition_files(source_paths: list[Path], partition_paths: list[Path], *, kind: str) -> None:
    if not partition_paths:
        return

    compressor = zstd.ZstdCompressor(level=OUTPUT_ZSTD_LEVEL)
    written = [0 for _ in partition_paths]
    skipped = 0

    with ExitStack() as stack:
        writers = []
        for path in partition_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fout = stack.enter_context(path.open("wb"))
            writers.append(stack.enter_context(compressor.stream_writer(fout, closefd=False)))

        for source_path in source_paths:
            log.info("Partitioning %s records from %s …", kind, source_path.name)
            for record in stream_zst(source_path):
                submission_id = _submission_id_for_record(record, kind)
                if not submission_id:
                    skipped += 1
                    continue
                partition_index = _partition_index(submission_id, len(partition_paths))
                writers[partition_index].write(_ORJSON_DUMPS(record) + b"\n")
                written[partition_index] += 1

    log.info(
        "Partitioned %s into %d shards (%d records written, %d skipped)",
        kind,
        len(partition_paths),
        sum(written),
        skipped,
    )


def _submission_id_for_record(record: dict[str, Any], kind: str) -> str | None:
    if kind == "submissions":
        submission_id = str(record.get("id", "")).strip()
        return submission_id or None

    link_id = str(record.get("link_id", "")).strip()
    if link_id.startswith("t3_"):
        return link_id[3:] or None

    parent_id = str(record.get("parent_id", "")).strip()
    if parent_id.startswith("t3_"):
        return parent_id[3:] or None

    fallback_id = str(record.get("id", "")).strip()
    return fallback_id or None


def _partition_index(submission_id: str, partitions: int) -> int:
    return zlib.crc32(submission_id.encode("utf-8")) % partitions
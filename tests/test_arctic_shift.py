"""Tests for src/arctic_shift.py — helper functions."""

from pathlib import Path

from src.arctic_shift import (
    MonthRef,
    _finalize_downloaded_files,
    _prioritize,
    _split_torrent_batches,
    build_magnet_uri,
    build_target_paths,
    iter_months,
)


def test_iter_months_single() -> None:
    months = iter_months((2022, 11), (2022, 11))
    assert months == [MonthRef(2022, 11)]


def test_iter_months_cross_year() -> None:
    months = iter_months((2022, 11), (2023, 2))
    assert len(months) == 4
    assert months[0] == MonthRef(2022, 11)
    assert months[-1] == MonthRef(2023, 2)


def test_build_target_paths_keys() -> None:
    paths = build_target_paths()
    assert "comments" in paths
    assert "submissions" in paths
    assert len(paths["comments"]) == len(paths["submissions"])


def test_build_target_paths_format() -> None:
    paths = build_target_paths()
    for p in paths["comments"]:
        assert p.startswith("reddit/comments/RC_")
        assert p.endswith(".zst")
    for p in paths["submissions"]:
        assert p.startswith("reddit/submissions/RS_")
        assert p.endswith(".zst")


def test_build_magnet_uri_contains_hash() -> None:
    uri = build_magnet_uri("abc123", ["udp://tracker.example.com:1234"])
    assert "xt=urn:btih:abc123" in uri
    assert "tr=udp://tracker.example.com:1234" in uri


def test_split_torrent_batches_uses_monthly_torrent_for_2024_02() -> None:
    batches = _split_torrent_batches(
        {
            "comments": ["reddit/comments/RC_2023-12.zst", "reddit/comments/RC_2024-02.zst"],
            "submissions": ["reddit/submissions/RS_2023-12.zst", "reddit/submissions/RS_2024-02.zst"],
        }
    )

    assert [batch.display_name for batch in batches] == [
        "reddit-2005-06-to-2023-12",
        "reddit-2024-02",
    ]
    assert batches[1].infohash == "5969ae3e21bb481fea63bf649ec933c222c1f824"
    assert batches[1].target_paths["comments"] == ["reddit/comments/RC_2024-02.zst"]
    assert batches[1].target_paths["submissions"] == ["reddit/submissions/RS_2024-02.zst"]


def test_prioritize_falls_back_to_basename_for_monthly_torrents() -> None:
    class FakeFiles:
        def num_files(self) -> int:
            return 2

    class FakeTorrentInfo:
        def files(self) -> FakeFiles:
            return FakeFiles()

    class FakeHandle:
        def __init__(self) -> None:
            self.priorities: list[int] | None = None

        def prioritize_files(self, priorities: list[int]) -> None:
            self.priorities = priorities

    handle = FakeHandle()
    selected, resolved = _prioritize(
        handle,
        FakeTorrentInfo(),
        {"RC_2024-02.zst": 0, "RS_2024-02.zst": 1},
        {
            "comments": ["reddit/comments/RC_2024-02.zst"],
            "submissions": ["reddit/submissions/RS_2024-02.zst"],
        },
    )

    assert selected == {"comments": [0], "submissions": [1]}
    assert handle.priorities == [4, 4]
    assert [item.actual_relpath for item in resolved] == ["RC_2024-02.zst", "RS_2024-02.zst"]


def test_finalize_downloaded_files_moves_flat_monthly_files(tmp_path: Path) -> None:
    actual_path = tmp_path / "RC_2024-02.zst"
    canonical_path = tmp_path / "reddit" / "comments" / "RC_2024-02.zst"
    actual_path.write_bytes(b"test-data")

    _finalize_downloaded_files({0: (actual_path, canonical_path)})

    assert canonical_path.exists()
    assert not actual_path.exists()
    assert canonical_path.with_suffix(".zst.complete").read_text() == str(canonical_path.stat().st_size)

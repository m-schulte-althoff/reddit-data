# reddit-data

Pipeline for extracting, filtering, and analysing Reddit comment and submission
data from the [Arctic Shift](https://github.com/ArcticalShift/arctic-shift)
archive (torrent bundle) or the
[Hugging Face mirror](https://huggingface.co/datasets/open-index/arctic).

## Quickstart

```bash
# 1. Download raw monthly .zst files (skips already-present files)
uv run python3 main.py download

# 2. Verify downloads are complete and valid
uv run python3 main.py verify

# 3. Filter raw data to the configured time window
uv run python3 main.py filter

# 4. Compute descriptive statistics
uv run python3 main.py analyse

# 5. Reservoir-sample 500 records per type to CSV
uv run python3 main.py sample
```

## Data sources

| Source | Method | Format |
|---|---|---|
| Arctic Shift torrent | BitTorrent (info-hash `9c263fc…`) | JSONL compressed with zstandard |
| Hugging Face `open-index/arctic` | DuckDB + `hf://` parquet reads | Parquet (ZSTD) |

The default pipeline uses the torrent source (`download` → `filter`).
The Hugging Face alternative is accessible via `hf-extract` / `hf-list`.

For Arctic Shift torrents, the downloader uses the legacy bundle through 2023-12
and switches to monthly torrents for 2024-01 onward.

## Configuration

Edit `src/config.py` to change:

- **`START_TS` / `END_EXCLUSIVE_TS`** — the exact UTC timestamp window.
  Month ranges and epoch values are derived automatically.
- **`SAMPLE_SIZE`** / **`RANDOM_SEED`** — sample parameters (default: 500 / 42).
- **`OUTPUT_ZSTD_LEVEL`** — compression level for filtered output.

## CLI commands

| Command | Description |
|---|---|
| `download` | Download missing raw `.zst` files via Arctic Shift torrent. |
| `verify` | Check that all raw files exist, are non-empty, and have valid zstd headers. |
| `filter` | Merge and filter raw files into `data/processed/` within the time window. |
| `analyse` | Compute descriptive stats (row counts, timestamp range, top subreddits/authors, score stats). |
| `sample` | Reservoir-sample records and write to `output/tables/`. |
| `hf-extract` | Extract filtered parquet from the Hugging Face mirror. |
| `hf-list` | List available months on the Hugging Face mirror. |

## File structure

```
├── main.py                  # CLI controller
├── views.py                 # Output formatting (CSV tables)
├── src/
│   ├── config.py            # Shared configuration
│   ├── arctic_shift.py      # Torrent download, filter, verify
│   ├── hugging_face.py      # HF parquet extraction
│   └── analysis.py          # Descriptive stats & sampling
├── data/
│   ├── raw/                 # Immutable raw .zst files (git-ignored)
│   └── processed/           # Filtered output (git-ignored)
├── output/
│   ├── tables/              # CSV summaries & samples
│   └── figures/             # (reserved for future plots)
├── logs/                    # Timestamped run logs
├── tests/
├── INSTRUCTIONS.md          # Code-style guidelines
└── README.md
```

## Reproducing results

```bash
# Full pipeline from scratch
uv run python3 main.py download
uv run python3 main.py verify
uv run python3 main.py filter
uv run python3 main.py analyse
uv run python3 main.py sample
```

All outputs are deterministic (fixed random seed, stable sorting).

## Development

```bash
uv run ruff check .
uv run pytest -q
```

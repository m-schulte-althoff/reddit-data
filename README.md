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

# 4. Filter by subreddit list (Chan-2025)
uv run python3 main.py filter-subreddit

# 5. If filtered data already exists: run the full filtered-data analysis stack
uv run python3 main.py all-analysis

# 6. Descriptive overview of filtered data (counts, trends)
uv run python3 main.py describe

# 7. Monthly subreddit panel for all downstream analyses
uv run python3 main.py panel

# 8. Main DiD / event-study analysis
uv run python3 main.py did

# 9. Responsiveness and support-availability metrics
uv run python3 main.py responsiveness

# 10. Moderator / mechanism analysis
uv run python3 main.py mechanisms

# 11. AI-mention trends
uv run python3 main.py ai-mentions

# 12. Simple content proxy metrics
uv run python3 main.py content-metrics

# 13. Bond-vs-identity interaction metrics
uv run python3 main.py interactions

# 14. Full first-pass WIP suite
uv run python3 main.py wip

# 15. Comment-depth / discursivity analysis
uv run python3 main.py discursivity

# 16. Engagement vs. post-GenAI decline analysis
uv run python3 main.py resilience

# 17. Repeat-helper concentration analysis
uv run python3 main.py helpers

# 18. Compute descriptive statistics on raw data
uv run python3 main.py analyse

# 19. Reservoir-sample 500 records per type to CSV
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
| `filter-subreddit` | Filter raw data by subreddit list (Chan-2025). Supports resume, time window, and month selection. |
| `all-analysis` | Run the full filtered-data analysis stack (`describe`, `discursivity`, `helpers`, `resilience`, `panel`, `did`, `responsiveness`, `mechanisms`, `ai-mentions`, `content-metrics`, `interactions`, `wip`) and write `output/summary.md`. |
| `panel` | Build the fingerprinted subreddit × month panel used by all downstream WIP analyses. |
| `did` | Estimate two-way fixed-effects DiD models, robustness checks, and event studies from the monthly panel. |
| `responsiveness` | Compute post-level and monthly responsiveness / support-availability metrics. |
| `mechanisms` | Test whether pre-period helper concentration, depth, and responsiveness moderate post-GenAI change. |
| `ai-mentions` | Count monthly mentions of ChatGPT, GPT, LLM, OpenAI, Copilot, Gemini, Claude, and related terms. |
| `content-metrics` | Compute lightweight effort / support / experience / information text proxies per subreddit-month. |
| `interactions` | Compute monthly bond-vs-identity interaction metrics: unique/repeat/new authors, dyads, thread concentration, and bond/identity indices. |
| `wip` | Run `panel` → `did` → `responsiveness` → `mechanisms` → `ai-mentions` → `content-metrics` → `interactions` and write concise key-result summaries. |
| `describe` | Descriptive overview of filtered data: post counts, monthly trends, per-subreddit breakdowns (CSV + SVG). |
| `discursivity` | Comment-depth / threading metrics from filtered data: mean depth, threading ratio per subreddit per month (CSV + SVG). |
| `resilience` | Engagement vs. post-GenAI decline: tests whether higher threading ratio / mean depth predicts less activity decline after ChatGPT launch. |
| `helpers` | Repeat-helper concentration: 1/9/90 shares, Top-1/5 share, HHI, Gini per subreddit per month; community-type comparison (general vs. health); moderation analysis. |
| `analyse` | Compute descriptive stats on raw data (row counts, timestamp range, top subreddits/authors, score stats). |
| `sample` | Reservoir-sample records and write to `output/tables/`. |
| `hf-extract` | Extract filtered parquet from the Hugging Face mirror. |
| `hf-list` | List available months on the Hugging Face mirror. |

## File structure

```
├── main.py                  # CLI controller
├── views.py                 # Output formatting (CSV, SVG figures)
├── src/
│   ├── config.py            # Shared configuration
│   ├── arctic_shift.py      # Torrent download, filter, verify
│   ├── filter.py            # Subreddit filtering with resume
│   ├── all_analysis.py      # Combined filtered-data analysis runner and summary
│   ├── io_utils.py          # Shared streaming, month, and fingerprint helpers
│   ├── describe.py          # Descriptive overview of filtered data
│   ├── panel.py             # Monthly subreddit panel with cache metadata
│   ├── did.py               # DiD / event-study models and robustness checks
│   ├── responsiveness.py    # Post-level responsiveness & monthly support metrics
│   ├── mechanisms.py        # Moderator analysis for resilience mechanisms
│   ├── ai_mentions.py       # Regex-based GenAI mention trends
│   ├── content_metrics.py   # Lightweight content / support proxy metrics
│   ├── interactions.py      # Bond-vs-identity interaction structure metrics
│   ├── discursivity.py      # Comment-depth & threading analysis
│   ├── resilience.py        # Engagement vs. post-GenAI decline analysis
│   ├── helpers.py           # Repeat-helper concentration analysis
│   ├── hugging_face.py      # HF parquet extraction
│   ├── wip.py               # Full WIP-suite orchestration and key results
│   └── analysis.py          # Descriptive stats & sampling (raw data)
├── input/
│   └── subreddit-list-Chan-2025.txt
├── data/
│   ├── raw/                 # Immutable raw .zst files (git-ignored)
│   └── processed/           # Filtered output (git-ignored)
├── output/
│   ├── summary.md           # Combined filtered-data analysis summary
│   ├── tables/              # CSV summaries & metrics
│   └── figures/             # SVG trend plots
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
uv run python3 main.py filter-subreddit

# If filtered data already exists, this one command runs the post-filter stack
uv run python3 main.py all-analysis

# Individual post-filter commands (optional / for debugging specific layers)
uv run python3 main.py panel
uv run python3 main.py did
uv run python3 main.py responsiveness
uv run python3 main.py mechanisms
uv run python3 main.py ai-mentions
uv run python3 main.py content-metrics
uv run python3 main.py interactions
uv run python3 main.py wip
uv run python3 main.py describe
uv run python3 main.py discursivity
uv run python3 main.py resilience
uv run python3 main.py helpers
uv run python3 main.py analyse
uv run python3 main.py sample
```

All outputs are deterministic (fixed random seed, stable sorting).

### All Analysis (filtered data already available)

If `data/processed/` already contains the filtered `.zst` files, the
`all-analysis` command is the shortest path to regenerate the post-filter
analysis stack.

```bash
uv run python3 main.py all-analysis
```

It runs, in order:

- `describe`
- `discursivity`
- `helpers`
- `resilience`
- `panel`
- `did`
- `responsiveness`
- `mechanisms`
- `ai-mentions`
- `content-metrics`
- `interactions`
- `wip`

Additional output:

| File | Content |
|---|---|
| `output/summary.md` | Combined Markdown index of the generated tables and figures, with the WIP key-results summary inlined at the top |

### Panel (monthly subreddit panel)

The `panel` command is the main WIP foundation. It streams filtered comments and
submissions, loads cached discursivity when valid, computes helper concentration
directly from comment-author counts, and writes a full subreddit × month grid.

Outputs:

| File | Content |
|---|---|
| `output/tables/community-monthly-panel.csv` | One row per subreddit-month with counts, author activity, text lengths, deleted/removed shares, score means, depth metrics, helper concentration, and post-GenAI timing fields |
| `output/tables/community-monthly-panel-metadata.json` | Input fingerprints, month coverage, row counts, and cache metadata |

### DiD and event study

The `did` command estimates fixed-effects models on the monthly panel with
subreddit-clustered standard errors. The summary table includes the baseline
unweighted model plus weighted, balanced-panel, winsorized, exclusion, and
alternative-cutoff robustness checks.

Outputs:

| File | Content |
|---|---|
| `output/tables/did-summary.csv` | Main DiD estimates and robustness rows for comments, submissions, and comments per submission |
| `output/tables/did-event-study-comments.csv` | Event-study coefficients for comment volume |
| `output/tables/did-event-study-submissions.csv` | Event-study coefficients for submission volume |
| `output/tables/did-event-study-comments-per-submission.csv` | Event-study coefficients for discussion intensity |
| `output/figures/did-trends-comments-health-vs-general.svg` | Monthly comment trends by community type |
| `output/figures/did-trends-submissions-health-vs-general.svg` | Monthly submission trends by community type |
| `output/figures/did-event-study-comments.svg` | Event-study coefficients for comments |
| `output/figures/did-event-study-submissions.svg` | Event-study coefficients for submissions |
| `output/figures/did-event-study-comments-per-submission.svg` | Event-study coefficients for comments per submission |

### Responsiveness and support availability

The `responsiveness` command uses a two-pass design: submissions are indexed
first, then comments update per-post reply timing, OP follow-up, commenter
diversity, and depth/threading metrics. Intermediate state is stored in
`output/cache/responsiveness.sqlite`.

Outputs:

| File | Content |
|---|---|
| `output/tables/responsiveness-posts.csv` | Post-level responsiveness metrics |
| `output/tables/responsiveness-monthly.csv` | Subreddit-month responsiveness summary |
| `output/figures/responsiveness-reply-rate-health-vs-general.svg` | Reply-rate trends |
| `output/figures/responsiveness-latency-health-vs-general.svg` | First-reply latency trends |
| `output/figures/responsiveness-op-followup-health-vs-general.svg` | OP follow-up trends |
| `output/figures/responsiveness-unanswered-rate-health-vs-general.svg` | Unanswered-rate trends |

### Mechanisms, AI mentions, content proxies, and interactions

These commands extend the WIP story beyond volume trends.

Outputs:

| Command | Main table outputs | Main figure outputs |
|---|---|---|
| `mechanisms` | `output/tables/mechanism-moderation-summary.csv` | `output/figures/mechanism-moderation-coefficients.svg`, `output/figures/mechanism-high-low-trends-top5.svg`, `output/figures/mechanism-high-low-trends-reply-rate.svg`, `output/figures/mechanism-high-low-trends-threading.svg` |
| `ai-mentions` | `output/tables/ai-mentions-monthly.csv` | `output/figures/ai-mentions-health-vs-general-comments.svg`, `output/figures/ai-mentions-health-vs-general-submissions.svg`, `output/figures/ai-mentions-top-subreddits.svg` |
| `content-metrics` | `output/tables/content-metrics-monthly.csv` | `output/figures/content-length-health-vs-general.svg`, `output/figures/content-question-share-health-vs-general.svg`, `output/figures/content-experience-share-health-vs-general.svg`, `output/figures/content-support-share-health-vs-general.svg` |
| `interactions` | `output/tables/interactions-monthly.csv` | `output/figures/interactions-bond-index-health-vs-general.svg`, `output/figures/interactions-identity-index-health-vs-general.svg` |

### WIP suite

The `wip` command runs the full first-pass manuscript workflow in the intended
order and reuses caches when the input fingerprints have not changed.

Outputs:

| File | Content |
|---|---|
| `output/tables/wip-key-results.csv` | Flat key-result summary for quick inspection or spreadsheet use |
| `output/tables/wip-key-results.md` | Markdown digest of coverage, DiD estimates, robustness checks, responsiveness, helper concentration, AI mentions, content proxies, and interaction indices |

### Describe (filtered data overview)

The `describe` command streams the filtered `.zst` files in `data/processed/`
and produces per-subreddit post counts and monthly trend graphs.

Outputs (per kind — comments and submissions):

| File | Content |
|---|---|
| `output/tables/describe-{kind}-summary.csv` | High-level stats + per-subreddit totals |
| `output/tables/describe-{kind}-monthly.csv` | Subreddit × month pivot table (all subreddits) |
| `output/tables/describe-{kind}-monthly-top15.csv` | Same, top 15 subreddits only |
| `output/figures/describe-{kind}-trend-aggregated.svg` | Aggregated monthly post volume |
| `output/figures/describe-{kind}-trend-community-types.svg` | Publication-ready monthly volume trend comparing healthcare vs. general communities |
| `output/figures/describe-{kind}-trend-all.svg` | Per-subreddit monthly trends (capped at 50 lines) |
| `output/figures/describe-{kind}-trend-top15.svg` | Top 15 subreddits monthly trends |

### Discursivity (comment depth & threading)

The `discursivity` command measures how deeply threaded discussions are across
subreddits. Each comment is assigned a depth (submission = 0, direct reply = 1,
reply-to-reply = 2, etc.) by resolving `parent_id` chains. Metrics are
aggregated per subreddit per month.

Key metrics:
- **Mean comment depth** — average depth of comments in a subreddit/month.
- **Threading ratio** — share of comments at depth ≥ 2 (replies to other
  comments, not directly to the submission).

Outputs:

| File | Content |
|---|---|
| `output/tables/discursivity-monthly.csv` | Long-format CSV: subreddit, month, comment/submission counts, mean depth, max depth, threading ratio, depth histogram |
| `output/figures/discursivity-mean-depth-top15.svg` | Mean depth trend, top 15 subreddits |
| `output/figures/discursivity-mean-depth-all.svg` | Mean depth trend, all subreddits (capped at 50) |
| `output/figures/discursivity-threading-ratio-top15.svg` | Threading ratio trend, top 15 subreddits |
| `output/figures/discursivity-threading-ratio-all.svg` | Threading ratio trend, all subreddits (capped at 50) |

The `discursivity` command also saves a **cache file**
(`output/tables/discursivity-cache.json`) that records the full result together
with SHA-256 fingerprints (filename, size, mtime) of the input `.zst` files.
Downstream commands like `resilience` load this cache automatically instead of
re-scanning the data — the cache is only reused when the input files are
unchanged; any modification triggers a recomputation.

### Resilience (engagement vs. post-GenAI decline)

The `resilience` command tests whether subreddits with deeper, more threaded
discussions experienced a smaller decline in monthly comment volume after the
ChatGPT launch (November 2022). It computes discursivity metrics, splits the
timeline at the GenAI cutoff, and runs both exploratory and rigorous analyses.

**Methodology:**
1. For each subreddit, compute pre-period (before 2022-11) and post-period
   average monthly comment counts and engagement metrics (threading ratio,
   mean comment depth) weighted by comment volume.
2. Calculate the percentage change in activity: `(post_mean − pre_mean) / pre_mean × 100`.
3. Subreddits with too few active months or too low activity are excluded
   (configurable thresholds: `min_pre_months=3`, `min_post_months=3`,
   `min_comments_per_month=10`).

**Exploratory outputs:**
- Scatter plots with OLS regression lines — one per engagement metric
- Indexed activity trend (pre-period = 100) for high vs. low threading groups
- Box plots comparing decline distributions by engagement level

**Rigorous statistical tests:**
- **Spearman rank correlation** (non-parametric) — tests monotonic association
  between pre-period engagement and post-period decline
- **OLS regression** (parametric) — quantifies the linear effect size
- **Mann–Whitney U test** (non-parametric) — compares decline distributions
  of above-median vs. below-median engagement groups

Outputs:

| File | Content |
|---|---|
| `output/tables/resilience-profiles.csv` | Per-subreddit profile: pre/post means, change %, pre-period threading ratio & depth |
| `output/tables/resilience-statistics.csv` | Spearman ρ, OLS coefficients, Mann–Whitney U statistics with p-values |
| `output/figures/resilience-scatter-threading.svg` | Scatter: pre-period threading ratio vs. activity change |
| `output/figures/resilience-scatter-depth.svg` | Scatter: pre-period mean depth vs. activity change |
| `output/figures/resilience-boxplot.svg` | Box plots: high vs. low engagement group decline (two panels) |
| `output/figures/resilience-indexed-trend.svg` | Indexed activity over time (high vs. low threading, with GenAI cutoff) |

## Filter by subreddit

The `filter-subreddit` command streams raw `.zst` files and keeps only records
whose `subreddit_name_prefixed` matches the list in
`input/subreddit-list-Chan-2025.txt`. Filtering is memory-safe (pure streaming),
resumable, and uses byte-level regex to avoid full JSON parsing of non-matching
lines.

```bash
# Filter all available raw data with the default subreddit list
uv run python3 main.py filter-subreddit

# Restrict to specific months
uv run python3 main.py filter-subreddit --months 2022-06,2023-06

# Restrict to a custom time window (ISO dates, start inclusive / end exclusive)
uv run python3 main.py filter-subreddit --start 2022-06-01 --end 2022-07-01

# Custom output tag and fresh run (no resume)
uv run python3 main.py filter-subreddit --tag chan2025 --no-resume
```

Output files are written to `data/processed/`:
- `filter-comments-<tag>.jsonl.zst`
- `filter-submissions-<tag>.jsonl.zst`

A sidecar `.progress.json` file tracks which input files have been processed.
On subsequent runs the already-completed files are skipped and new data is
appended as additional zstd frames.

## Development

```bash
uv run ruff check .
uv run --extra dev pytest -q
```

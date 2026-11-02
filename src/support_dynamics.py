"""Describe helper replenishment from event dates and capacity relative to demand."""

from __future__ import annotations

import json
import logging
import pickle
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from src.config import OUTPUT_DIR, TABLES_DIR
from src.helpers import classify_subreddit
from src.io_utils import extract_created_utc, fingerprint_hash, fingerprint_paths, stream_zst

log = logging.getLogger(__name__)


def _observe(authors: dict[str, int], author: str, month: int, width: int, *,
             helper: bool = False, posting: bool = False) -> None:
    """Pack activity, helper months, and first posting month into one integer."""
    author = author.strip()
    if not author or author.casefold() in {"[deleted]", "[removed]"}:
        return
    # Month bitsets keep millions of author histories in memory without storing event rows.
    state = authors.get(author, 0) | (1 << month)
    if helper:
        state |= 1 << (width + month)
    if posting:
        first_post = state >> (2 * width)
        if first_post == 0 or month + 1 < first_post:
            state = (state & ((1 << (2 * width)) - 1)) | ((month + 1) << (2 * width))
    authors[author] = state


def collect_histories(submission_paths: list[Path], comment_paths: list[Path],
                      months: list[str]) -> dict[str, dict[str, int]]:
    """Read matching thread partitions once; date helping by comment timestamps."""
    if len(submission_paths) != len(comment_paths) or not submission_paths:
        raise ValueError("Matching nonempty submission and comment partitions are required")
    boundaries = [int(datetime.fromisoformat(m + '-01').replace(tzinfo=timezone.utc).timestamp()) for m in months]
    last = datetime.fromisoformat(months[-1] + '-01')
    end = datetime(last.year + (last.month == 12), last.month % 12 + 1, 1, tzinfo=timezone.utc)
    boundaries.append(int(end.timestamp()))
    width = len(months)
    histories: dict[str, dict[str, int]] = {}
    for partition, (submission_path, comment_path) in enumerate(zip(submission_paths, comment_paths, strict=True), 1):
        threads: dict[str, tuple[str, str]] = {}
        current_month = 0
        for record in stream_zst(submission_path):
            subreddit = str(record.get('subreddit', ''))
            if classify_subreddit(subreddit) not in {'health', 'general'}:
                continue
            submission_id = str(record.get('id', ''))
            timestamp = extract_created_utc(record)
            if not submission_id or timestamp is None or not boundaries[0] <= timestamp < boundaries[-1]:
                continue
            author = str(record.get('author', ''))
            threads[submission_id] = (subreddit, author)
            if not boundaries[current_month] <= timestamp < boundaries[current_month + 1]:
                current_month = bisect_right(boundaries, timestamp) - 1
            _observe(histories.setdefault(subreddit, {}), author, current_month, width, posting=True)
        current_month = 0
        for record in stream_zst(comment_path):
            link_id = str(record.get('link_id', ''))
            if not link_id.startswith('t3_') or (thread := threads.get(link_id[3:])) is None:
                continue
            if not record.get('id'):
                continue
            timestamp = extract_created_utc(record)
            if timestamp is None or not boundaries[0] <= timestamp < boundaries[-1]:
                continue
            if not boundaries[current_month] <= timestamp < boundaries[current_month + 1]:
                current_month = bisect_right(boundaries, timestamp) - 1
            subreddit, op = thread
            author = str(record.get('author', ''))
            _observe(histories[subreddit], author, current_month, width, helper=author != op)
        log.info('Replenishment scan partition %d/%d; %s author histories',
                 partition, len(submission_paths), f'{sum(map(len, histories.values())):,}')
    return histories


def summarize_histories(histories: dict[str, dict[str, int]], months: list[str]) -> pd.DataFrame:
    """Count disjoint helper sources and newcomer-poster conversion by cohort."""
    width = len(months)
    mask = (1 << width) - 1
    rows = []
    for subreddit, authors in sorted(histories.items()):
        counts: Counter[tuple[int, str]] = Counter()
        for state in authors.values():
            activity = state & mask
            helpers = (state >> width) & mask
            first_seen = (activity & -activity).bit_length() - 1
            first_post = (state >> (2 * width)) - 1
            if first_post == first_seen:
                counts[(first_seen, 'newcomer_posters')] += 1
                if not helpers & (1 << first_seen):
                    counts[(first_seen, 'poster_only_newcomers')] += 1
                    for horizon in (1, 3):
                        future = ((1 << horizon) - 1) << (first_seen + 1)
                        if helpers & future:
                            counts[(first_seen, f'converted_{horizon}m')] += 1
            remaining = activity
            while remaining:
                bit = remaining & -remaining
                counts[(bit.bit_length() - 1, 'active_contributors')] += 1
                remaining ^= bit
            remaining = helpers
            while remaining:
                bit = remaining & -remaining
                index = bit.bit_length() - 1
                if helpers & (bit >> 1):
                    category = 'continuing_helpers'
                elif helpers & (bit - 1):
                    category = 'reactivated_helpers'
                elif index == first_seen:
                    category = 'newcomer_helpers'
                else:
                    category = 'converting_helpers'
                counts[(index, category)] += 1
                remaining ^= bit
        for index, month in enumerate(months):
            row: dict[str, Any] = {'subreddit': subreddit, 'month': month, 'community_type': classify_subreddit(subreddit)}
            for name in ('active_contributors', 'newcomer_posters', 'poster_only_newcomers', 'continuing_helpers',
                         'reactivated_helpers', 'newcomer_helpers', 'converting_helpers'):
                row[name] = counts[(index, name)]
            total = sum(row[name] for name in ('continuing_helpers', 'reactivated_helpers',
                                               'newcomer_helpers', 'converting_helpers'))
            row['event_active_helpers'] = total
            row['event_helper_return_1m'] = (
                counts[(index + 1, 'continuing_helpers')] / total
                if total and index + 1 < width else float('nan'))
            for name in ('continuing', 'reactivated', 'newcomer', 'converting'):
                row[f'{name}_helper_share'] = row[f'{name}_helpers'] / total if total else float('nan')
            for horizon in (1, 3):
                observed = index + horizon < width and row['poster_only_newcomers'] > 0
                row[f'poster_to_helper_{horizon}m'] = (
                    counts[(index, f'converted_{horizon}m')] / row['poster_only_newcomers']
                    if observed else float('nan'))
            row['history_observed'] = index >= 6
            rows.append(row)
    return pd.DataFrame(rows)


def _collect_chunk(number: int, submission_paths: list[Path], comment_paths: list[Path],
                   months: list[str], cache: Path) -> Path:
    """Checkpoint one bounded worker so an interrupted scan can reuse finished chunks."""
    path = cache / f'histories-{number}.pickle'
    if not path.exists():
        histories = collect_histories(submission_paths, comment_paths, months)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    return path


def summarize_replenishment_years(events: pd.DataFrame) -> pd.DataFrame:
    """Match calendar cohort months across years for each future horizon."""
    frame = events[events.history_observed.eq(True)].copy()
    frame['year'] = frame.month.str[:4]
    keys = ['community_type', 'year']
    community_keys = ['subreddit', *keys]
    columns = ['continuing_helper_share', 'reactivated_helper_share', 'newcomer_helper_share',
               'converting_helper_share', 'poster_to_helper_1m', 'poster_to_helper_3m', 'event_helper_return_1m']
    summary = frame.groupby(community_keys)[columns].mean().groupby(keys).mean()
    for column, final_month in [('poster_to_helper_1m', '11'), ('poster_to_helper_3m', '09'),
                                ('event_helper_return_1m', '11')]:
        matched = frame[frame.month.str[-2:].le(final_month)]
        summary[column] = matched.groupby(community_keys)[column].mean().groupby(keys).mean()
    return summary


def _merge_histories(target: dict[str, dict[str, int]], source: dict[str, dict[str, int]], width: int) -> None:
    """Union month flags and preserve the earliest posting month across thread shards."""
    mask = (1 << (2 * width)) - 1
    for subreddit, authors in source.items():
        destination = target.setdefault(subreddit, {})
        for author, right in authors.items():
            left = destination.get(author, 0)
            first_left, first_right = left >> (2 * width), right >> (2 * width)
            first = min(first_left, first_right) if first_left and first_right else first_left or first_right
            destination[author] = ((left | right) & mask) | (first << (2 * width))


def build_demand_panel(panel: pd.DataFrame, support: pd.DataFrame,
                       responsiveness: pd.DataFrame) -> pd.DataFrame:
    """Align effective capacity and submissions within complete 12-month windows."""
    frame = support.merge(panel[['subreddit', 'month', 'submissions', 'comments']],
                          on=['subreddit', 'month'], validate='one_to_one')
    frame = frame.merge(responsiveness[['subreddit', 'month', 'reply_rate', 'mean_non_op_commenters']],
                        on=['subreddit', 'month'], validate='one_to_one')
    frame = frame.sort_values(['subreddit', 'month']).copy()
    frame['rolling_submissions_12m'] = frame.groupby('subreddit').submissions.transform(
        lambda values: values.rolling(12, min_periods=12).sum())
    frame['effective_helpers_per_1000_submissions_12m'] = (
        1000 * frame.rolling_effective_helpers / frame.rolling_submissions_12m.replace(0, float('nan')))
    return frame


def run_support_dynamics_analysis() -> None:
    """Extend the audited tables with event-time replenishment and demand measures."""
    panel = pd.read_csv(TABLES_DIR / 'community-monthly-panel.csv')
    support = pd.read_csv(TABLES_DIR / 'support-capacity-monthly.csv')
    responsiveness = pd.read_csv(TABLES_DIR / 'responsiveness-monthly.csv')
    months = sorted(panel.month.unique())
    root = OUTPUT_DIR / 'cache/thread-prep/submission-hash-0064'
    manifest = json.loads((root / 'thread-prep-manifest.json').read_text())
    panel_sources = json.loads((TABLES_DIR / 'interactions-metadata.json').read_text())['sources']
    for key in ('comment_files', 'submission_files'):
        if manifest['sources'][key] != panel_sources[key]:
            raise ValueError('Thread partitions do not match the rebuilt interaction sources')
    submissions = sorted((root / 'submissions').glob('*.jsonl.zst'))
    comments = sorted((root / 'comments').glob('*.jsonl.zst'))
    if len(submissions) != 64 or len(comments) != 64:
        raise ValueError('The audited set of 64 thread partitions is required')
    metadata_path = TABLES_DIR / 'support-dynamics-metadata.json'
    events_path = TABLES_DIR / 'support-dynamics-replenishment-monthly.csv'
    sources = fingerprint_paths(submissions + comments)
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if events_path.exists() and previous.get('partition_sources') == sources and previous.get('version') == 3:
        log.info('Using verified event-time replenishment cache')
        events = pd.read_csv(events_path)
    else:
        # shortcut: checkpoints assume this packed-history format; bump v1 if its format changes.
        cache = OUTPUT_DIR / 'cache/support-dynamics' / ('v1-' + fingerprint_hash(sources))
        cache.mkdir(parents=True, exist_ok=True)
        histories: dict[str, dict[str, int]] = {}
        # Three processes parallelize parsing while keeping author histories within workstation memory.
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_collect_chunk, number, submissions[number::3],
                       comments[number::3], months, cache) for number in range(3)]
            for future in as_completed(futures):
                with future.result().open('rb') as handle:
                    partial = pickle.load(handle)
                _merge_histories(histories, partial, len(months))
                del partial
        events = summarize_histories(histories, months)
        expected = support[support.community_type.isin(['health', 'general'])][
            ['subreddit', 'month', 'active_contributors']]
        check = events.merge(expected, on=['subreddit', 'month'], suffixes=('', '_expected'), validate='one_to_one')
        if len(check) != len(expected) or not check.active_contributors.eq(check.active_contributors_expected).all():
            raise ValueError('Event-time participant counts disagree with the audited interaction cache')
        events.to_csv(events_path, index=False)
        metadata_path.write_text(json.dumps({'version': 3, 'partition_sources': sources,
            'audit_sources': panel_sources, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'participant_counts_match_audited_cache': True,
            'helper_month': 'actual comment timestamp; non-OP comments on observed submissions',
            'conversion': 'poster-only in first-observed month; first non-OP comment in the next 1 or 3 calendar months',
            'history_warmup_months': 6}, indent=2))
    demand = build_demand_panel(panel, support, responsiveness)
    summarize_replenishment_years(events).to_csv(TABLES_DIR / 'support-dynamics-replenishment-summary.csv')
    demand.to_csv(TABLES_DIR / 'support-dynamics-demand-monthly.csv', index=False)
    selected = demand[demand.community_type.isin(['health', 'general'])].copy()
    selected['year'] = selected.month.str[:4]
    metrics = ['comments', 'submissions', 'rolling_effective_helpers', 'rolling_top5_helper_share',
               'effective_helpers_per_1000_submissions_12m', 'newcomer_reply_rate', 'reply_rate', 'mean_non_op_commenters']
    annual = selected.groupby(['subreddit', 'community_type', 'year'], as_index=False)[metrics].mean()
    annual.to_csv(TABLES_DIR / 'support-dynamics-community-year.csv', index=False)
    early = annual[annual.year.eq('2023')].drop(columns='year')
    late = annual[annual.year.eq('2025')].drop(columns='year')
    changes = early.merge(late, on=['subreddit', 'community_type'], suffixes=('_2023', '_2025'), validate='one_to_one')
    for metric in metrics:
        changes[f'{metric}_change'] = changes[f'{metric}_2025'] - changes[f'{metric}_2023']
        changes[f'{metric}_change_pct'] = 100 * (
            changes[f'{metric}_2025'] / changes[f'{metric}_2023'].replace(0, float('nan')) - 1)
    changes.to_csv(TABLES_DIR / 'support-dynamics-community-changes.csv', index=False)
    association_rows = []
    for group, frame in changes.groupby('community_type'):
        for x in ('rolling_effective_helpers_change_pct', 'effective_helpers_per_1000_submissions_12m_change_pct'):
            pair = frame[[x, 'newcomer_reply_rate_change']].dropna()
            rho, p = spearmanr(pair[x], pair.newcomer_reply_rate_change)
            association_rows.append({'community_type': group, 'x': x,
                'y': 'newcomer_reply_rate_change', 'n': len(pair), 'spearman_rho': rho, 'p_value': p})
    pd.DataFrame(association_rows).to_csv(TABLES_DIR / 'support-dynamics-associations.csv', index=False)
    log.info('Wrote replenishment, demand, community changes, and descriptive associations')

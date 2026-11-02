"""Check role transitions, censoring, and aligned demand windows."""

import json
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from src.support_dynamics import (
    _merge_histories, _observe, build_demand_panel, collect_histories,
    summarize_histories, summarize_replenishment_years,
)


def test_conversion_years_match_calendar_cohorts() -> None:
    frame = pd.DataFrame([
        {'subreddit': 'Anxiety', 'community_type': 'health', 'month': month, 'history_observed': True,
         'continuing_helper_share': .5, 'reactivated_helper_share': .2,
         'newcomer_helper_share': .2, 'converting_helper_share': .1,
         'poster_to_helper_1m': value, 'poster_to_helper_3m': value, 'event_helper_return_1m': value}
        for month, value in [('2023-09', .2), ('2023-10', .8), ('2023-12', 1), ('2025-09', .2)]
    ])
    means = summarize_replenishment_years(frame)
    assert means.loc[('health', '2023'), 'poster_to_helper_3m'] == .2
    assert means.loc[('health', '2023'), 'poster_to_helper_1m'] == .5
    assert means.loc[('health', '2025'), 'poster_to_helper_3m'] == .2
    extra = frame.iloc[[1]].assign(subreddit='Asthma', month='2023-09')
    balanced = summarize_replenishment_years(pd.concat([frame, extra], ignore_index=True))
    assert balanced.loc[('health', '2023'), 'poster_to_helper_1m'] == pytest.approx(.65)


def test_merge_preserves_events_and_earliest_post() -> None:
    left, right = {}, {}
    _observe(left, 'person', 2, 5, posting=True)
    _observe(right, 'person', 0, 5, posting=True)
    _observe(right, 'person', 1, 5, helper=True)
    _merge_histories({'Anxiety': left}, {'Anxiety': right}, 5)
    frame = summarize_histories({'Anxiety': left}, [f'2025-0{i}' for i in range(1, 6)])
    assert frame.loc[0, 'newcomer_posters'] == 1
    assert frame.loc[0, 'poster_to_helper_1m'] == 1
    assert frame.loc[1, 'converting_helpers'] == 1


def test_roles_conversion_and_censoring() -> None:
    months = [f'2025-{m:02d}' for m in range(1, 9)]
    authors = {}
    _observe(authors, 'new', 0, 8, posting=True)
    _observe(authors, 'new', 1, 8, helper=True)
    _observe(authors, 'new', 2, 8, helper=True)
    _observe(authors, 'new', 4, 8, helper=True)
    _observe(authors, 'immediate', 1, 8, posting=True, helper=True)
    _observe(authors, 'not-yet', 1, 8, posting=True)
    _observe(authors, 'prior-commenter', 0, 8)
    _observe(authors, 'prior-commenter', 1, 8, posting=True)
    _observe(authors, 'prior-commenter', 2, 8, helper=True)
    _observe(authors, 'late', 7, 8, posting=True)
    _observe(authors, 'boundary', 4, 8, posting=True)
    _observe(authors, 'boundary', 7, 8, helper=True)
    _observe(authors, 'censored', 5, 8, posting=True)
    _observe(authors, 'censored', 7, 8, helper=True)
    _observe(authors, '[deleted]', 1, 8, helper=True)
    frame = summarize_histories({'Anxiety': authors}, months).set_index('month')
    assert frame.loc['2025-01', 'poster_to_helper_1m'] == 1
    assert frame.loc['2025-01', 'poster_to_helper_3m'] == 1
    assert frame.loc['2025-02', 'newcomer_posters'] == 2
    assert frame.loc['2025-02', 'poster_only_newcomers'] == 1
    assert frame.loc['2025-02', 'poster_to_helper_1m'] == 0
    assert frame.loc['2025-02', 'newcomer_helpers'] == 1
    assert frame.loc['2025-02', 'converting_helpers'] == 1
    assert frame.loc['2025-03', 'continuing_helpers'] == 1
    assert frame.loc['2025-02', 'event_helper_return_1m'] == pytest.approx(.5)
    assert pd.isna(frame.loc['2025-08', 'event_helper_return_1m'])
    assert frame.loc['2025-05', 'reactivated_helpers'] == 1
    assert pd.isna(frame.loc['2025-08', 'poster_to_helper_1m'])
    assert pd.isna(frame.loc['2025-06', 'poster_to_helper_3m'])
    assert frame.loc['2025-05', 'poster_to_helper_3m'] == 1
    assert frame.loc['2025-06', 'poster_to_helper_1m'] == 0
    categories = ['continuing_helpers', 'reactivated_helpers', 'newcomer_helpers', 'converting_helpers']
    assert frame[categories].sum(axis=1).eq(frame.event_active_helpers).all()


def test_partition_collection_uses_comment_date_and_excludes_op(tmp_path: Path) -> None:
    def write(name, records):
        path = tmp_path / name
        data = ''.join(json.dumps(record) + '\n' for record in records).encode()
        path.write_bytes(zstd.ZstdCompressor().compress(data))
        return path
    def ts(month):
        return int(pd.Timestamp(month + '-15', tz='UTC').timestamp())
    submissions = write('submissions.zst', [
        {'id': 'p', 'subreddit': 'Anxiety', 'author': 'op', 'created_utc': ts('2025-01')},
        {'id': 'q', 'subreddit': 'Anxiety', 'author': 'other', 'created_utc': ts('2025-02')},
    ])
    comments = write('comments.zst', [
        {'id': 'c', 'link_id': 't3_p', 'author': 'op', 'created_utc': ts('2025-02')},
        {'id': 'd', 'link_id': 't3_q', 'author': 'op', 'created_utc': ts('2025-03')},
        {'id': 'e', 'link_id': 't3_missing', 'author': 'ignored', 'created_utc': ts('2025-03')},
    ])
    months = ['2025-01', '2025-02', '2025-03', '2025-04']
    histories = collect_histories([submissions], [comments], months)
    assert 'ignored' not in histories['Anxiety']
    frame = summarize_histories(histories, months).set_index('month')
    assert frame.loc['2025-01', 'event_active_helpers'] == 0
    assert frame.loc['2025-02', 'event_active_helpers'] == 0
    assert frame.loc['2025-03', 'converting_helpers'] == 1
    assert frame.loc['2025-01', 'poster_to_helper_1m'] == 0
    assert frame.loc['2025-01', 'poster_to_helper_3m'] == 1


def test_demand_requires_twelve_matching_months() -> None:
    months = pd.date_range('2024-01-01', periods=13, freq='MS').strftime('%Y-%m')
    panel = pd.DataFrame({'subreddit': 'Anxiety', 'month': months, 'submissions': 10, 'comments': 20})
    support = panel[['subreddit', 'month']].assign(rolling_effective_helpers=12)
    response = panel[['subreddit', 'month']].assign(reply_rate=.8, mean_non_op_commenters=2)
    result = build_demand_panel(panel, support, response)
    assert result.effective_helpers_per_1000_submissions_12m.iloc[:11].isna().all()
    assert result.rolling_submissions_12m.iloc[-1] == 120
    assert result.effective_helpers_per_1000_submissions_12m.iloc[-1] == pytest.approx(100)

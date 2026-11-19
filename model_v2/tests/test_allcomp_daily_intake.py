from copy import deepcopy
from datetime import date, timedelta

import pytest

from football_model.allcomp_daily_intake import (
    coverage_intervals, merge_parsed_events, validate_daily_payload,
)
from football_model.all_competition_features import (
    AllCompetitionFeatures, ScheduleClub, ScheduleEvent,
)


def parsed(identifier='one', day='2026-10-19', **changes):
    return {'event_id': identifier, 'match_date': day, 'home_team_id': '1',
            'away_team_id': '2', 'competition_slug': 'eng.1',
            'completed': True, 'category': 'competitive',
            'is_regular_league_fixture': True, 'exclusion_reason': None, **changes}


def test_event_boundary_deduplication_and_conflicts_never_overwrite():
    first = parsed(source_final_score=[1, 0])
    duplicate = parsed(source_final_score=[99, 99])
    conflict = parsed('rescheduled', '2026-10-18')
    moved = parsed('rescheduled', '2026-10-20', away_team_id='3',
                   competition_slug='eng.fa', completed=False,
                   category='unclassified', is_regular_league_fixture=False)
    rows = [first, duplicate, conflict, moved]
    original = deepcopy(rows)
    result = merge_parsed_events(rows)
    assert rows == original
    assert [row['event_id'] for row in result['events']] == ['one']
    assert result['conflict_ids'] == ['rescheduled']
    assert result['conflict_dates'] == ['2026-10-18', '2026-10-20']
    assert result['conflicts'][0]['team_ids'] == ['1', '2', '3']
    assert set(result['conflicts'][0]['fields']) == {
        'match_date', 'away_team_id', 'competition_slug', 'completed',
        'category', 'is_regular_league_fixture'}
    reverse = merge_parsed_events(reversed(rows))
    assert reverse['conflicts'] == result['conflicts']
    cutoff = date(2026, 10, 26)
    events = [ScheduleEvent(row['event_id'], date.fromisoformat(row['match_date']),
              row['competition_slug'], row['category'], row['completed'],
              row['is_regular_league_fixture'])
              for row in merge_parsed_events([first, duplicate, parsed('cutoff', cutoff)])['events']]
    source = AllCompetitionFeatures([ScheduleClub('A', 'E0', '1',
        coverage_intervals(cutoff-timedelta(days=n) for n in range(1, 29)), events)])
    features, metadata = source._snapshot('E0', 'A', cutoff)
    assert features['formal_count_7d'] == 1.
    assert metadata['windows']['7']['selected_event_ids'] == ['one']


def test_http_success_shape_wrong_league_day_and_truncation_are_not_observed_zero():
    day = date(2026, 10, 19)
    event = {'id': '1', 'date': '2026-10-19T23:30Z'}
    valid = {'leagues': [{'slug': 'eng.1'}], 'events': [event]}
    assert validate_daily_payload(valid, 'eng.1', day) == [event]
    assert validate_daily_payload({'leagues': [{'slug': 'eng.1'}], 'events': []}, 'eng.1', day) == []
    bad = [
        {}, {'leagues': [{'slug': 'eng.1'}], 'events': {}},
        {**valid, 'leagues': [{'slug': 'eng.fa'}]},
        {**valid, 'events': [{'date': '2026-10-20T00:00Z'}]},
        {**valid, 'events': [{'date': '2026-10-19T00:30+02:00'}]},
        {**valid, 'events': [{'date': '2026-10-19T20:00'}]},
        {**valid, 'events': [{**event, 'league': {'slug': 'eng.fa'}}]},
        {**valid, 'pagination': {'total': 2}},
        {**valid, 'pageInfo': {'hasNextPage': True}},
        {**valid, 'links': [{'rel': ['next'], 'href': 'another-page'}]},
    ]
    for payload in bad:
        with pytest.raises(ValueError):
            validate_daily_payload(payload, 'eng.1', day)
    with pytest.raises(ValueError, match='limit'):
        validate_daily_payload(valid, 'eng.1', day, limit=1)


def test_partial_coverage_retains_usable_last_seven_days_but_not_twenty_eight():
    cutoff = date(2026, 10, 26)
    days = {cutoff-timedelta(days=n) for n in range(1, 29)}
    missing = cutoff-timedelta(days=12)
    days.remove(missing)
    intervals = coverage_intervals([*days, min(days)])
    assert intervals == [(cutoff-timedelta(days=28), missing),
                         (missing+timedelta(days=1), cutoff)]
    assert coverage_intervals([]) == []
    event = ScheduleEvent('cup', cutoff-timedelta(days=3),
                          'uefa.europa.conf_qual', 'competitive', True, False)
    source = AllCompetitionFeatures([ScheduleClub('A', 'E0', '1', intervals, [event])])
    f, _ = source._snapshot('E0', 'A', cutoff)
    assert f['window_usable_7d'] == 1.
    assert f['formal_count_7d'] == f['nonleague_count_7d'] == 1.
    assert f['window_usable_28d'] == 0.
    assert f['formal_count_28d'] is None
    assert f['formal_count_14d'] is None
    assert f['covered_days_within_28'] == 11.

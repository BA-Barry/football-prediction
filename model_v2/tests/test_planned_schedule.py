from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from football_model.domain import Fixture
from football_model.planned_schedule import planned_schedule_context


AS_OF = datetime(2026, 10, 27, 8, tzinfo=timezone.utc)
TARGET = Fixture('target', 'A', 'B', 'E0',
                 datetime(2026, 10, 31, 15, tzinfo=timezone.utc))
SCOPE = {'competition': 'eng.league_cup', 'days': ['2026-10-27', '2026-10-29']}


def event(identifier='cup', **changes):
    return {'event_id': identifier, 'home': 'A', 'away': 'C',
            'kickoff': '2026-10-28T19:45:00Z', 'competition': 'eng.league_cup',
            'status': 'scheduled', 'observed_at': '2026-10-27T07:00:00Z',
            'source_url': 'https://example.test/fixtures', **changes}


def context(events, **changes):
    return planned_schedule_context(TARGET, events, as_of=AS_OF, source_scope=SCOPE, **changes)


def test_target_later_unrelated_future_observation_and_duplicate_do_not_enter_plan():
    rows = [event(), event(),
            event('target', kickoff=TARGET.kickoff.isoformat(), away='B'),
            event('different-provider-target', kickoff='2026-10-31T16:00:00+01:00', home='B', away='A'),
            event('later', kickoff='2026-11-01T20:00:00Z'),
            event('other', home='D', away='C'),
            event('not-yet-observed', observed_at='2026-10-27T08:00:01Z')]
    before, scope_before = deepcopy(rows), deepcopy(SCOPE)
    result = context(rows)
    assert rows == before and SCOPE == scope_before
    assert [r['event_id'] for r in result['home']['scheduled_between_as_of_and_target']] == ['cup']
    assert result['home']['scheduled_fixture_count_before_target'] == 1
    assert result['away']['scheduled_fixture_count_before_target'] == 0
    assert result['away']['nearest_scheduled_kickoff_to_target_hours'] is None
    assert result['ignored_counts'] == {
        'observed_after_as_of': 1, 'target_itself': 2,
        'at_or_after_target': 1, 'unrelated_teams': 1, 'duplicate_id': 1}
    assert result['applied_to_statistical_rates'] is False
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_actual_completion_is_separate_from_plans_and_kickoff_gap_is_timezone_exact():
    rows = [event(kickoff='2026-10-28T20:45:00+01:00', away='B'),
            event('finished', kickoff='2026-10-25T15:00Z', status='completed'),
            event('ongoing', kickoff='2026-10-27T07:00Z', status='in_progress'),
            event('stale-plan', kickoff='2026-10-27T07:30Z'),
            event('exact-boundary', kickoff=AS_OF.isoformat()),
            event('impossible-completion', kickoff='2026-10-28T20:00Z', status='completed'),
            event('delayed', status='postponed'), event('cancelled', status='canceled')]
    result = context(rows)
    own = result['home']
    plan = own['scheduled_between_as_of_and_target']
    assert len(plan) == 1 and plan[0]['scheduled_kickoff_to_target_hours'] == 67.25
    assert result['away']['scheduled_between_as_of_and_target'] == plan
    assert own['nearest_scheduled_kickoff_to_target_hours'] == 67.25
    assert [r['event_id'] for r in own['completed_before_as_of']] == ['finished']
    assert {r['event_id'] for r in own['unresolved_before_target']} == {
        'ongoing', 'stale-plan', 'exact-boundary', 'impossible-completion'}
    assert {r['event_id'] for r in own['cancelled_or_postponed_before_target']} == {'delayed', 'cancelled'}
    for bucket in ('scheduled_between_as_of_and_target', 'completed_before_as_of',
                   'unresolved_before_target', 'cancelled_or_postponed_before_target'):
        assert all('actual_rest_hours' not in row and 'rotation_probability' not in row
                   and 'strength_adjustment' not in row for row in own[bucket])


def test_invalid_time_and_conflicting_schedule_ids_are_not_silently_interpreted():
    with pytest.raises(ValueError, match='timezone-aware'):
        context([event(kickoff='2026-10-28T19:45:00')])
    with pytest.raises(ValueError, match='timezone-aware'):
        context([event(observed_at='2026-10-27T07:00:00')])
    with pytest.raises(ValueError, match='before target'):
        planned_schedule_context(TARGET, [], as_of=TARGET.kickoff, source_scope=SCOPE)
    with pytest.raises(ValueError, match='Conflicting schedule versions'):
        context([event(), event(kickoff='2026-10-29T19:45:00Z')])

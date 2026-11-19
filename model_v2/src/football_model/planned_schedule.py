"""Describe known schedules before a target, without estimating fatigue/rotation.

This pure adapter keeps planned kickoffs separate from observed completion. Its
hour differences are kickoff-to-kickoff, never actual recovery time. It does not
alter a statistical forecast or invent probabilities for unknown player usage.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone

from .domain import Fixture, aware


def _timestamp(value, name):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be an aware ISO timestamp string')
    try:
        instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError(f'{name} must be an aware ISO timestamp string') from error
    return aware(instant).astimezone(timezone.utc)


def planned_schedule_context(target: Fixture, events: list[dict], *,
                             as_of: datetime, source_scope: dict):
    """Return per-side facts available at ``as_of`` and before the target.

    Event fields: event_id, home, away, kickoff (aware ISO), competition, status,
    observed_at (aware ISO), source_url. Canonical team strings must match the
    target's team identities. Duplicate IDs with the same schedule/status count
    once; contradictory versions are returned to the caller as a ValueError,
    rather than arbitrarily selecting a version. No target outcomes are used.

    An empty list means no qualifying record in source_scope, not proof of a
    club's complete all-competition schedule. Publication/revision times are not
    inferred from observed_at. Target and later events are deliberately omitted.
    """
    aware(as_of)
    if as_of >= target.kickoff:
        raise ValueError('Schedule context must be generated before target kickoff')
    if not isinstance(source_scope, Mapping):
        raise ValueError('Source scope must describe the actual requested coverage')
    contexts = {
        side: {'team': team, 'scheduled_between_as_of_and_target': [],
               'completed_before_as_of': [], 'unresolved_before_target': [],
               'cancelled_or_postponed_before_target': []}
        for side, team in (('home', target.home), ('away', target.away))
    }
    ignored = {'observed_after_as_of': 0, 'target_itself': 0,
               'at_or_after_target': 0, 'unrelated_teams': 0, 'duplicate_id': 0}
    selected = {}
    fact_fields = ('home', 'away', 'kickoff', 'competition', 'status')
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError('Schedule events must be mappings')
        row = {}
        for field in ('event_id', 'home', 'away', 'competition', 'status', 'source_url'):
            if not isinstance(event.get(field), str) or not event[field].strip():
                raise ValueError(f'Schedule event needs explicit {field}')
            row[field] = event[field]
        if row['home'] == row['away']:
            raise ValueError('Schedule event teams must differ')
        observed = _timestamp(event.get('observed_at'), 'observed_at')
        if observed > as_of:
            ignored['observed_after_as_of'] += 1
            continue
        kickoff = _timestamp(event.get('kickoff'), 'kickoff')
        row.update(kickoff=kickoff.isoformat(), observed_at=observed.isoformat(),
                   status=row['status'].strip().lower())
        if (row['event_id'] == target.match_id
                or (kickoff == target.kickoff
                    and {row['home'], row['away']} == {target.home, target.away})):
            ignored['target_itself'] += 1
            continue
        if kickoff >= target.kickoff:
            ignored['at_or_after_target'] += 1
            continue
        if not {row['home'], row['away']} & {target.home, target.away}:
            ignored['unrelated_teams'] += 1
            continue
        previous = selected.get(row['event_id'])
        if previous is not None:
            if any(previous[field] != row[field] for field in fact_fields):
                raise ValueError(f'Conflicting schedule versions for event {row["event_id"]}')
            ignored['duplicate_id'] += 1
            continue
        selected[row['event_id']] = row
    for row in sorted(selected.values(), key=lambda r: (r['kickoff'], r['event_id'])):
        kickoff = _timestamp(row['kickoff'], 'kickoff')
        description = {**row, 'scheduled_kickoff_to_target_hours':
                       (target.kickoff-kickoff).total_seconds()/3600}
        status = row['status']
        if status == 'scheduled' and kickoff > as_of:
            bucket = 'scheduled_between_as_of_and_target'
        elif status == 'completed' and kickoff < as_of:
            bucket = 'completed_before_as_of'
        elif status in {'cancelled', 'canceled', 'postponed'}:
            bucket = 'cancelled_or_postponed_before_target'
        else:
            bucket = 'unresolved_before_target'
        for context in contexts.values():
            if context['team'] in (row['home'], row['away']):
                context[bucket].append(deepcopy(description))
    for context in contexts.values():
        planned = context['scheduled_between_as_of_and_target']
        context['scheduled_fixture_count_before_target'] = len(planned)
        context['nearest_scheduled_kickoff_to_target_hours'] = min(
            (row['scheduled_kickoff_to_target_hours'] for row in planned), default=None)
    return {'target_fixture_id': target.match_id,
            'target_kickoff': target.kickoff.astimezone(timezone.utc).isoformat(),
            'as_of': as_of.astimezone(timezone.utc).isoformat(),
            'source_scope': deepcopy(dict(source_scope)), **contexts,
            'ignored_counts': ignored,
            'applied_to_statistical_rates': False,
            'limitations': [
                'Kickoff-to-kickoff hours are not actual rest, travel duration or measured fatigue.',
                'Planned fixtures are not completed load; delays, extra time and participant minutes are unknown.',
                'Starting lineups, rotation, tactics and competition priority have no assigned probabilities or strength adjustments.',
                'Empty lists describe the supplied source scope, not certified all-competition absence.',
                'Observed time does not establish first publication or historical revision availability.']}

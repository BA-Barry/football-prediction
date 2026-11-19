"""Pure adapters for daily all-competition schedule responses.

No request, file access, model fit or score interpretation happens here. A valid
response establishes only the requested provider/day scope; it does not certify
that the provider contains every football competition. HTTP status and which
competition/day requests succeeded remain the caller's responsibility.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone


def _calendar_day(value):
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError('Need an explicit calendar date')


def _event_utc_day(event):
    value = event.get('date')
    if not isinstance(value, str):
        raise ValueError('Daily events need an explicit source timestamp')
    try:
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid daily event timestamp') from error
    if timestamp.tzinfo is None:
        raise ValueError('Daily event timestamp needs a time zone')
    return timestamp.astimezone(timezone.utc).date()


def _check_pagination(container, count):
    """Reject visible evidence that this response is only part of its scope."""
    if not isinstance(container, Mapping):
        raise ValueError('Malformed response pagination')
    for key in ('hasMore', 'hasNext', 'hasNextPage', 'next', 'nextPage',
                'nextCursor', 'continuationToken'):
        if container.get(key) not in (None, False, '', 0):
            raise ValueError('Daily response advertises additional pages')
    for key in ('total', 'totalCount', 'totalEvents'):
        value = container.get(key)
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError('Malformed response total')
            if value > count:
                raise ValueError('Daily response contains fewer events than its total')
    for key in ('pageCount', 'totalPages', 'page', 'pageIndex'):
        value = container.get(key)
        if value is not None and (type(value) is not int or value < 0 or value > 1):
            raise ValueError('Daily response is paginated')
    for key in ('offset', 'startIndex'):
        value = container.get(key)
        if value is not None and (type(value) is not int or value != 0):
            raise ValueError('Daily response has a nonzero page offset')
    links = container.get('links', [])
    if isinstance(links, list):
        for link in links:
            if isinstance(link, Mapping):
                rel = link.get('rel', [])
                rel = [rel] if isinstance(rel, str) else rel
                if isinstance(rel, list) and 'next' in rel and link.get('href'):
                    raise ValueError('Daily response links to a next page')


def validate_daily_payload(payload, slug, day: date, limit=1000):
    """Return raw events after checking one requested UTC day and league.

    HTTP 200 alone is insufficient: missing events, a different league/day, an
    advertised extra page or a response at the request limit is not usable as an
    observed zero. The runner catches ValueError and retains that day as missing.
    Empty events is valid when league identity and the remaining checks pass.
    """
    if type(day) is not date or not isinstance(slug, str) or not slug:
        raise ValueError('Need a requested league and date-only UTC day')
    if type(limit) is not int or limit < 1:
        raise ValueError('Need a positive event limit')
    if not isinstance(payload, Mapping) or not isinstance(payload.get('events'), list):
        raise ValueError('Daily response must contain an events list')
    leagues = payload.get('leagues')
    if (not isinstance(leagues, list) or not leagues
            or any(not isinstance(league, Mapping) or league.get('slug') != slug
                   for league in leagues)):
        raise ValueError('Daily response league identity does not match the request')
    events = payload['events']
    if len(events) >= limit:
        raise ValueError('Daily response reaches the event limit; truncation is possible')
    _check_pagination(payload, len(events))
    for key in ('pagination', 'paging', 'pageInfo'):
        if key in payload:
            _check_pagination(payload[key], len(events))
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError('Daily response contains a malformed event')
        if _event_utc_day(event) != day:
            raise ValueError('Daily event UTC date differs from the requested day')
        league = event.get('league')
        if league is not None and (not isinstance(league, Mapping)
                                   or league.get('slug') not in (None, slug)):
            raise ValueError('Daily event league identity differs from the request')
    return events


_IDENTITY_FIELDS = ('match_date', 'home_team_id', 'away_team_id',
                    'competition_slug', 'completed', 'category',
                    'is_regular_league_fixture', 'exclusion_reason')


def merge_parsed_events(rows: Iterable[Mapping]):
    """Deduplicate parsed IDs without silently picking a conflicting version.

    Returns events, conflict_ids, conflict_dates and detailed conflicts. Conflicted
    IDs are absent from events. The caller marks their involved club/days unknown;
    dropping those IDs alone must never be interpreted as a measured zero.
    Scores, display names and other non-feature source metadata do not participate
    in identity comparison. All inputs remain unchanged.
    """
    by_id = defaultdict(list)
    for original in rows:
        if not isinstance(original, Mapping):
            raise ValueError('Parsed events must be mappings')
        row = deepcopy(dict(original))
        for field in ('event_id', 'home_team_id', 'away_team_id', 'competition_slug', 'category'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Parsed event needs explicit {field}')
        if row['home_team_id'] == row['away_team_id']:
            raise ValueError('Parsed event sides must be different')
        if type(row.get('completed')) is not bool:
            raise ValueError('Parsed event needs an explicit completion flag')
        if ('is_regular_league_fixture' in row
                and type(row['is_regular_league_fixture']) is not bool):
            raise ValueError('Parsed regular-league flag must be boolean')
        if row.get('exclusion_reason') is not None and not isinstance(row['exclusion_reason'], str):
            raise ValueError('Parsed exclusion reason must be text or null')
        row['match_date'] = _calendar_day(row.get('match_date')).isoformat()
        by_id[row['event_id']].append(row)
    events, conflicts = [], []
    for identifier, versions in sorted(by_id.items()):
        fields = [field for field in _IDENTITY_FIELDS
                  if any(row.get(field) != versions[0].get(field) for row in versions[1:])]
        if fields:
            conflicts.append({'event_id': identifier, 'fields': fields,
                'match_dates': sorted({row['match_date'] for row in versions}),
                'team_ids': sorted({row[side] for row in versions
                                    for side in ('home_team_id', 'away_team_id')})})
        else:
            events.append(versions[0])
    return {'events': sorted(events, key=lambda row: (row['match_date'], row['event_id'])),
            'conflict_ids': [row['event_id'] for row in conflicts],
            'conflict_dates': sorted({day for row in conflicts for day in row['match_dates']}),
            'conflicts': conflicts}


def coverage_intervals(days: Iterable[date]):
    """Turn successful UTC dates into contiguous [start, end) date intervals.

    Supply only days whose required competition responses were usable and which
    were not invalidated by relevant unknown/conflicting events. This function
    does not infer success from the presence or absence of an event.
    """
    days = list(days)
    if any(type(day) is not date for day in days):
        raise ValueError('Coverage days must be date-only values')
    result = []
    for day in sorted(set(days)):
        end = day + timedelta(days=1)
        if result and result[-1][1] == day:
            result[-1] = (result[-1][0], end)
        else:
            result.append((day, end))
    return result

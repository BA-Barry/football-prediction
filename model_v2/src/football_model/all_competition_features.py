"""Date-only, strictly earlier all-competition schedule observations.

Completed competitive event counts are not measured playing minutes or causal
fatigue. Coverage means the club's requested historical member-season scope,
not a claim that the provider contains every competition. Publication times and
rescheduling history are unavailable. Future fixture records are never inputs.
"""
from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path


LEAGUE_SLUGS = {'E0': 'eng.1', 'D1': 'ger.1', 'SP1': 'esp.1',
                'I1': 'ita.1', 'F1': 'fra.1'}
WINDOWS = (7, 14, 28)


@dataclass
class ScheduleEvent:
    event_id: str
    match_date: date
    competition_slug: str
    category: str
    completed: bool
    is_regular_league_fixture: bool
    exclusion_reason: str | None = None


@dataclass
class ScheduleClub:
    team: str
    competition: str
    source_team_id: str
    coverage: list[tuple[date, date]]
    events: list[ScheduleEvent]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _merged_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if type(start) is not date or type(end) is not date or start >= end:
            raise ValueError('Need nonempty calendar coverage intervals')
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


class AllCompetitionFeatures:
    """Build earlier date/count observations with explicit source uncertainty.

    Windows are relative to the model's cutoff, not the target date. Gaps are
    log1p(days), separating last observed kickoff to cutoff from forecast lead.
    Counts become missing if their window is not covered or contains a known
    partial/resumed/uncertain competitive event. No actual minutes are assumed.
    """

    def __init__(self, clubs, *, source_metadata=None):
        self._clubs, self._cache = {}, {}
        self.source_metadata = source_metadata or {}
        for club in clubs:
            if (not club.team or not club.source_team_id or club.competition not in LEAGUE_SLUGS
                    or (club.competition, club.team) in self._clubs):
                raise ValueError('Need unique explicit club identities')
            coverage = _merged_intervals(club.coverage)
            events, ids = [], set()
            for event in sorted(club.events, key=lambda e: (e.match_date, e.event_id)):
                if type(event.match_date) is not date or not event.event_id or event.event_id in ids:
                    raise ValueError('Need unique calendar-dated event identities')
                if not isinstance(event.completed, bool) or not isinstance(event.is_regular_league_fixture, bool):
                    raise ValueError('Need explicit source completion and regular-league flags')
                ids.add(event.event_id)
                events.append(event)
            self._clubs[(club.competition, club.team)] = ScheduleClub(
                club.team, club.competition, club.source_team_id, coverage, events)

    @classmethod
    def from_directory(cls, directory):
        directory = Path(directory)
        seasons = _read(directory/'四季覆盖汇总.json')['seasons']
        blocked, by_club, sources = {}, {}, []
        for season in seasons:
            summary = _read(season['summary_file'])
            for row in summary.get('known_specials', []):
                blocked[str(row['event_id'])] = row['reason']
        official = directory/'官方部分比赛补充说明_20261025.json'
        if official.exists():
            for row in _read(official)['matches']:
                blocked[str(row['event_id'])] = 'officially_confirmed_partial_match'
        for season in seasons:
            year = int(season['season'].split('-')[0])
            path = Path(season['events_file']).with_name('相邻赛季补全_逐队历史观察.json')
            payload = _read(path)
            sources.append(str(path))
            for source_id, row in payload['clubs'].items():
                comp = row['competition']
                team = f"football-data:{comp}:{row['independent_label']}"
                key = (comp, team)
                current = by_club.setdefault(key, {'id': str(source_id), 'coverage': [], 'events': {}})
                if current['id'] != str(source_id) or row['team_id'] != str(source_id):
                    raise ValueError('Historical member identities disagree across seasons')
                current['coverage'].append((date(year, 7, 1), date(year+1, 7, 1)))
                for event in row['events']:
                    event_id = str(event['event_id'])
                    when = datetime.fromisoformat(event['date_utc_text'].replace('Z', '+00:00')).date()
                    if not date(year, 7, 1) <= when < date(year+1, 7, 1):
                        raise ValueError('Member-season events exceed the stated calendar scope')
                    if str(source_id) not in (event['home_team_id'], event['away_team_id']):
                        raise ValueError('Historical member is absent from its event')
                    reason = blocked.get(event_id)
                    if event['category'] == 'competitive' and event['completed'] is not True:
                        reason = reason or 'unresolved_source_completion_status'
                    clean = ScheduleEvent(event_id, when, event['competition_slug'], event['category'],
                        event['completed'], event['is_regular_league_fixture'], reason)
                    previous = current['events'].get(event_id)
                    if previous is not None and previous != clean:
                        raise ValueError('Duplicate historical event changes feature fields')
                    current['events'][event_id] = clean
        clubs = [ScheduleClub(team, comp, row['id'], row['coverage'], list(row['events'].values()))
                 for (comp, team), row in by_club.items()]
        return cls(clubs, source_metadata={
            'directory': str(directory), 'source_files': sources,
            'club_competition_identities': len(clubs), 'club_seasons': sum(len(c.coverage) for c in clubs),
            'blocked_event_ids_and_reasons': blocked,
            'scope': 'historical member-season requested schedule coverage, not certified competition completeness',
            'source_first_publication_unknown': True, 'uses_scores_or_lineups': False})

    def _snapshot(self, competition, team, cutoff):
        cache_key = (competition, team, cutoff)
        if cache_key in self._cache:
            return self._cache[cache_key]
        club = self._clubs.get((competition, team))
        interval = next(((start, end) for start, end in club.coverage if start <= cutoff <= end), None) if club else None
        prior = [r for r in club.events if interval and interval[0] <= r.match_date < cutoff
                 and r.category == 'competitive'] if club else []
        good = [r for r in prior if r.completed and r.exclusion_reason is None]
        unknown = [r for r in prior if not r.completed or r.exclusion_reason is not None]
        features = {
            'identity_available': float(club is not None),
            'scope_contains_cutoff': float(interval is not None),
            'covered_days_within_28': float(min(28, (cutoff-interval[0]).days)) if interval else 0.,
        }
        metadata = {'team': team, 'source_team_id': club.source_team_id if club else None,
            'contiguous_member_scope': [d.isoformat() for d in interval] if interval else None,
            'cutoff': cutoff.isoformat(), 'windows': {}}
        for days in WINDOWS:
            left = date.fromordinal(cutoff.toordinal()-days)
            covered = interval is not None and interval[0] <= left
            selected = [r for r in good if left <= r.match_date]
            issues = [r for r in unknown if left <= r.match_date]
            usable = covered and not issues
            features[f'window_usable_{days}d'] = float(usable)
            features[f'formal_count_{days}d'] = float(len(selected)) if usable else None
            features[f'nonleague_count_{days}d'] = float(sum(
                r.competition_slug != LEAGUE_SLUGS[competition] or not r.is_regular_league_fixture
                for r in selected)) if usable else None
            features[f'unresolved_count_{days}d'] = float(len(issues)) if covered else None
            metadata['windows'][str(days)] = {'covered': covered, 'usable': usable,
                'selected_event_ids': [r.event_id for r in selected],
                'selected_event_dates': [r.match_date.isoformat() for r in selected],
                'excluded_or_unresolved_events': [{'event_id': r.event_id, 'date': r.match_date.isoformat(),
                    'reason': r.exclusion_reason} for r in issues]}
        last = good[-1] if good else None
        last_usable = last is not None and not any(r.match_date >= last.match_date for r in unknown)
        features['last_log1p_gap_to_cutoff_days'] = math.log1p((cutoff-last.match_date).days) if last_usable else None
        features['last_is_nonleague'] = float(last.competition_slug != LEAGUE_SLUGS[competition]
            or not last.is_regular_league_fixture) if last_usable else None
        recent = good[-3:]
        recent_usable = len(recent) == 3 and not any(r.match_date >= recent[0].match_date for r in unknown)
        features['previous3_count'] = float(len(recent)) if interval else None
        features['previous3_usable'] = float(recent_usable)
        features['previous3_log1p_mean_interval_days'] = math.log1p(sum(
            (b.match_date-a.match_date).days for a, b in zip(recent, recent[1:]))/2) if recent_usable else None
        metadata['last_completed_event'] = {'event_id': last.event_id, 'date': last.match_date.isoformat(),
            'competition_slug': last.competition_slug, 'usable': last_usable} if last else None
        metadata['previous3_events'] = [{'event_id': r.event_id, 'date': r.match_date.isoformat()} for r in recent]
        metadata['previous3_usable'] = recent_usable
        result = features, metadata
        self._cache[cache_key] = result
        return result

    def for_match(self, *, home, away, competition, cutoff, match_date):
        if (type(cutoff) is not date or type(match_date) is not date or cutoff > match_date
                or competition not in LEAGUE_SLUGS or not home or not away or home == away):
            raise ValueError('Need valid date-only forecast identities and cutoff no later than target')
        own, own_meta = self._snapshot(competition, home, cutoff)
        other, other_meta = self._snapshot(competition, away, cutoff)
        lead = math.log1p((match_date-cutoff).days)
        home_features = {**{f'own_allcomp_{k}': v for k, v in own.items()},
                         **{f'opponent_allcomp_{k}': v for k, v in other.items()},
                         'allcomp_log1p_forecast_lead_days': lead}
        away_features = {**{f'own_allcomp_{k}': v for k, v in other.items()},
                         **{f'opponent_allcomp_{k}': v for k, v in own.items()},
                         'allcomp_log1p_forecast_lead_days': lead}
        return {'home_features': home_features, 'away_features': away_features,
            'metadata': {'cutoff': cutoff.isoformat(), 'match_date': match_date.isoformat(),
                'history': {'home': own_meta, 'away': other_meta},
                'chronology': 'source calendar day strictly before cutoff; all windows end at cutoff',
                'interpretation': 'recorded event counts / kickoff dates, not minutes or latest target-day rest; future fixtures unused'}}

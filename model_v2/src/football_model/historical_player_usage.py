"""Earlier appearances -> team usage inputs, without a target squad or XI.

Understat ``time`` is a provider minute counter, not an exact on-pitch clock.
Use its observed team total for shares, never repair that total to 990. These
inputs describe past domestic-league usage, not injuries, present availability,
physiological fatigue, player ability, or the target match's expected minutes.
"""

from collections import defaultdict
from datetime import date, timedelta
from math import fsum, isfinite


def extract_usage(payload, fixture):
    """Read only IDs, side and source minutes from an independently joined game.

    The caller supplies the existing verified fixture join. No score, shot,
    position, starter flag, substitution, or individual performance is read.
    """
    played = date.fromisoformat(fixture['date'])
    rosters = payload['rosters']
    observations, seen = [], set()
    for source_side, side in (('h', 'home'), ('a', 'away')):
        minutes, clubs = {}, set()
        for row in rosters[source_side].values():
            player = str(row['player_id'])
            club = str(row['team_id'])
            value = float(row['time'])
            if (not player.isdecimal() or not club.isdecimal()
                    or int(player) <= 0 or int(club) <= 0
                    or row['h_a'] != source_side or player in seen
                    or not isfinite(value) or not 0 <= value <= 90):
                raise ValueError('Inconsistent historical appearance identity or source minutes')
            seen.add(player)
            clubs.add(club)
            minutes[f'understat:player:{player}'] = value
        if len(clubs) != 1 or not minutes or fsum(minutes.values()) <= 0:
            raise ValueError('Need one observed source club and positive team minutes')
        observations.append({
            'match_id': fixture['match_id'], 'date': played.isoformat(),
            'team_id': fixture[side], 'source_club_id': clubs.pop(),
            'competition': fixture['competition'], 'season': fixture['season'],
            'source_url': fixture['source_url'], 'minutes': minutes,
            'source_team_minutes': fsum(minutes.values()),
        })
    return observations


def _shares(records):
    totals = defaultdict(float)
    for row in records:
        for player, value in row['minutes'].items():
            totals[player] += value
    total = fsum(totals.values())
    return {player: value / total for player, value in totals.items() if value > 0}


def _overlap(a, b):
    return fsum(min(value, b.get(player, 0.)) for player, value in a.items())


def usage_features(team_history, *, team_id, before_date):
    """Describe strictly earlier history using fixed, interpretable windows.

    Five recent fixtures and the preceding five describe usage changes. A
    28-calendar-day history supplies reference shares for previous-seven-day
    recorded workload. Windows are descriptive presets, not tuned parameters.
    A missing comparison returns None; it is never interpreted as zero change.
    Earlier-date availability is a research assumption, not a verified original
    publication timestamp. Same-day games are excluded in their entirety.
    """
    if type(before_date) is not date:
        raise ValueError('Use a calendar-date information cutoff')
    earlier = sorted((row for row in team_history if row['team_id'] == team_id
                      and date.fromisoformat(row['date']) < before_date),
                     key=lambda row: (row['date'], row['match_id']))
    if len({row['match_id'] for row in earlier}) != len(earlier):
        raise ValueError('Duplicate historical team appearances')
    recent, preceding = earlier[-5:], earlier[-10:-5]
    recent_shares, preceding_shares = _shares(recent), _shares(preceding)
    previous_28 = [row for row in earlier
                   if date.fromisoformat(row['date']) >= before_date - timedelta(days=28)]
    previous_7 = [row for row in previous_28
                  if date.fromisoformat(row['date']) >= before_date - timedelta(days=7)]
    reference = _shares(previous_28)
    weekly_minutes = defaultdict(float)
    for row in previous_7:
        for player, value in row['minutes'].items():
            weekly_minutes[player] += value
    consecutive = [_overlap(_shares([a]), _shares([b]))
                   for a, b in zip(recent, recent[1:])]
    features = {
        'usage_history_matches': len(earlier),
        'usage_recent_matches': len(recent),
        'usage_comparison_matches': len(preceding),
        'usage_days_since_last': (before_date-date.fromisoformat(earlier[-1]['date'])).days if earlier else None,
        'usage_effective_players_recent5': 1 / fsum(p*p for p in recent_shares.values()) if recent else None,
        'usage_top11_share_recent5': fsum(sorted(recent_shares.values(), reverse=True)[:11]) if recent else None,
        'usage_adjacent_overlap_recent5': fsum(consecutive)/len(consecutive) if consecutive else None,
        'usage_shift_vs_previous5': 1-_overlap(recent_shares, preceding_shares) if recent and preceding else None,
        'usage_matches_previous7days': len(previous_7),
        'usage_matches_previous28days': len(previous_28),
        'usage_reference_weighted_minutes_previous7days': fsum(reference[p]*weekly_minutes[p] for p in reference) if reference else None,
    }
    used = sorted({row['match_id']: row for row in recent + preceding + previous_28}.values(),
                  key=lambda row: (row['date'], row['match_id']))
    return {
        'team_id': team_id, 'before_date': before_date.isoformat(),
        'features': features,
        'history_match_ids': [row['match_id'] for row in used],
        'latest_history_date': earlier[-1]['date'] if earlier else None,
        'history_available': bool(earlier),
        'current_squad_known': False,
        'source_minutes_semantics': 'Understat reported minutes; shares use observed source totals',
        'coverage': 'EPL matches in supplied archive only; cups, Europe, internationals and training absent',
        'historical_publication_time_known': False,
    }

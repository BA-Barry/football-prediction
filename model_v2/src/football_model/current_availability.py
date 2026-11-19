"""Pure comparison of two official FPL bootstrap observations.

Source labels/news are observations, not medical diagnoses, confirmed lineups,
estimated player effects or calibrated appearance probabilities. Pairing uses
stable source team/player codes, never names or season-local numeric IDs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy

from .prospective import _time


_PLAYER_FIELDS = ('id', 'code', 'first_name', 'second_name', 'web_name',
                  'element_type', 'status', 'news', 'news_added',
                  'chance_of_playing_this_round', 'chance_of_playing_next_round',
                  'minutes', 'starts')
_CHANCE_FIELDS = ('chance_of_playing_this_round', 'chance_of_playing_next_round')


def _identifier(row, key, label):
    value = row.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f'{label} needs a positive source {key}')
    return value


def _snapshot(payload, label):
    if (not isinstance(payload, Mapping) or not isinstance(payload.get('teams'), list)
            or not isinstance(payload.get('elements'), list)):
        raise ValueError(f'{label} needs raw bootstrap teams/elements lists')
    by_id, teams = {}, {}
    for raw in payload['teams']:
        if not isinstance(raw, Mapping):
            raise ValueError('Team rows must be mappings')
        identifier = _identifier(raw, 'id', 'Team')
        code = _identifier(raw, 'code', 'Team')
        if identifier in by_id or code in teams:
            raise ValueError(f'Duplicate team source id/code in {label}')
        by_id[identifier] = code
        teams[code] = {key: deepcopy(raw.get(key)) for key in ('id', 'code', 'name', 'short_name')}
    players, player_codes, player_ids = {}, {}, set()
    for raw in payload['elements']:
        if not isinstance(raw, Mapping):
            raise ValueError('Player rows must be mappings')
        identifier = _identifier(raw, 'id', 'Player')
        code = _identifier(raw, 'code', 'Player')
        team_id = _identifier(raw, 'team', 'Player')
        if identifier in player_ids or code in player_codes:
            raise ValueError(f'Duplicate player source id/code in {label}')
        if team_id not in by_id:
            raise ValueError('Player source team id is absent from this snapshot')
        team_code = by_id[team_id]
        row = {key: deepcopy(raw.get(key)) for key in _PLAYER_FIELDS}
        row.update(team_id=team_id, team_code=team_code,
                   source_status_class={'a': 'available_source_label',
                       'i': 'injury_source_label', 'd': 'doubtful_source_label',
                       's': 'suspended_source_label', 'u': 'unavailable_other_source_label'
                       }.get(row['status'], 'unknown_source_status'))
        if row['status'] is not None and not isinstance(row['status'], str):
            raise ValueError('Player source status must be text or null')
        # Validate parseability only. news_added is retained as raw source text;
        # it is never promoted to injury onset or first-publication time.
        if row['news_added'] is not None:
            _time(row['news_added'], 'news_added source metadata')
        players[(team_code, code)] = row
        player_codes[code] = team_code
        player_ids.add(identifier)
    return teams, players, player_codes


def _metadata_instant(value):
    return None if value is None else _time(value, 'news_added source metadata')


def compare_fpl_snapshots(current, earlier, *, current_observed_at, earlier_observed_at):
    """Return current team descriptions and separately typed source changes.

    Inputs are raw bootstrap dictionaries plus aware ISO observation timestamps.
    Only the same (team code, player code) is compared across observations. A
    transfer, new player or removed player remains unpaired; identical names and
    reused local IDs never establish a comparison. Raw chance fields are kept as
    source percentages, not converted to estimated appearance/starting chances.
    """
    now = _time(current_observed_at, 'current_observed_at')
    before = _time(earlier_observed_at, 'earlier_observed_at')
    if before > now:
        raise ValueError('Earlier snapshot cannot be observed after current snapshot')
    current_teams, current_players, current_codes = _snapshot(current, 'current')
    earlier_teams, earlier_players, earlier_codes = _snapshot(earlier, 'earlier')
    paired = sorted(current_players.keys() & earlier_players.keys())
    status_changes, news_changes, metadata_changes, chance_changes = [], [], [], []
    for key in paired:
        old, new = earlier_players[key], current_players[key]
        identity = {'team_code': key[0], 'player_code': key[1],
                    'earlier_id': old['id'], 'current_id': new['id'],
                    'current_web_name': new['web_name']}
        if old['status'] != new['status']:
            status_changes.append({**identity, 'before': old['status'], 'after': new['status']})
        if old['news'] != new['news']:
            news_changes.append({**identity, 'before': old['news'], 'after': new['news']})
        if _metadata_instant(old['news_added']) != _metadata_instant(new['news_added']):
            metadata_changes.append({**identity, 'before': old['news_added'], 'after': new['news_added']})
        changed = {field: {'before': old[field], 'after': new[field]}
                   for field in _CHANCE_FIELDS if old[field] != new[field]}
        if changed:
            chance_changes.append({**identity, 'fields': changed})

    def unpaired(players, other, other_codes):
        output = []
        for key in sorted(players.keys()-other.keys()):
            code_elsewhere = other_codes.get(key[1])
            output.append({'player': deepcopy(players[key]),
                'reason': 'same_player_code_different_team' if code_elsewhere is not None
                          else 'not_in_other_snapshot',
                'other_snapshot_team_code': code_elsewhere})
        return output

    teams = []
    for code, team in sorted(current_teams.items()):
        players = [deepcopy(row) for key, row in sorted(current_players.items()) if key[0] == code]
        counts = Counter(row['status'] if row['status'] is not None else '<null_or_missing>'
                         for row in players)
        teams.append({'team': deepcopy(team), 'earlier_team': deepcopy(earlier_teams.get(code)),
            'current_roster_count': len(players), 'current_status_counts': dict(sorted(counts.items())),
            'current_players': players,
            'flagged_players': [deepcopy(row) for row in players if row['status'] in {'i', 'd', 's'}],
            'unavailable_other_players': [deepcopy(row) for row in players if row['status'] == 'u'],
            'unknown_status_players': [deepcopy(row) for row in players
                                       if row['source_status_class'] == 'unknown_source_status']})
    return {'current_observed_at': now.isoformat(), 'earlier_observed_at': before.isoformat(),
        'pairing_key': ['source_team_code', 'source_player_code'], 'teams': teams,
        'paired_player_count': len(paired), 'status_changes': status_changes,
        'news_changes': news_changes, 'news_metadata_changes': metadata_changes,
        'chance_field_changes': chance_changes,
        'unpaired_current_players': unpaired(current_players, earlier_players, earlier_codes),
        'unpaired_earlier_players': unpaired(earlier_players, current_players, current_codes),
        'limits': [
            'Status changes are changes between source snapshots, not new injury incidence or verified diagnoses.',
            'Status u is other unavailability/registration/transfer metadata, not an injury label.',
            'Status a is a source label, not proof of health, participation or a starting place; null/unknown remains unknown.',
            'news_added is source metadata, not injury onset or independently established first publication.',
            'Raw chance fields are source percentage labels, not calibrated appearance or starting probabilities.',
            'Minutes and starts are cumulative source counters, not ability, future minutes, fatigue or a strength adjustment.',
            'Transfers/new/removed players remain unpaired; names and local IDs are not cross-snapshot identity evidence.',
            'Observation times are caller-supplied source receipt claims; this function performs no network, model fit or forecast.']}

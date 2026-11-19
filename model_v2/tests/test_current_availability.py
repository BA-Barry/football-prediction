from copy import deepcopy
import json

import pytest

from football_model.current_availability import compare_fpl_snapshots


BEFORE = '2026-10-26T08:00:00Z'
CURRENT = '2026-10-27T08:00:00+00:00'


def player(identifier=1, code=101, team=1, **changes):
    return {'id': identifier, 'code': code, 'team': team, 'first_name': 'Same',
            'second_name': 'Name', 'web_name': 'Same Name', 'element_type': 3,
            'status': 'a', 'news': '', 'news_added': None,
            'chance_of_playing_this_round': None, 'chance_of_playing_next_round': None,
            'minutes': 400, 'starts': 5, **changes}


def snapshot(players=None):
    return {'teams': [{'id': 1, 'code': 10, 'name': 'Team A', 'short_name': 'A'},
                      {'id': 2, 'code': 20, 'name': 'Team B', 'short_name': 'B'}],
            'elements': players if players is not None else [player()]}


def compare(current, earlier):
    return compare_fpl_snapshots(current, earlier, current_observed_at=CURRENT,
                                 earlier_observed_at=BEFORE)


def test_source_states_news_metadata_and_chance_changes_are_separate():
    earlier = snapshot([player(status='d', news='Training assessment pending',
                               news_added='2026-10-25T10:00:00Z',
                               chance_of_playing_next_round=50)])
    current = snapshot([player(status='i', news='Ruled out by source',
                               news_added='2026-10-26T10:00:00Z',
                               chance_of_playing_next_round=0, minutes=490, starts=6)])
    originals = deepcopy((current, earlier))
    result = compare(current, earlier)
    assert (current, earlier) == originals
    assert result['status_changes'][0]['before'] == 'd'
    assert result['status_changes'][0]['after'] == 'i'
    assert result['news_changes'][0]['after'] == 'Ruled out by source'
    assert len(result['news_metadata_changes']) == len(result['chance_field_changes']) == 1
    flagged = result['teams'][0]['flagged_players'][0]
    assert flagged['chance_of_playing_next_round'] == 0
    assert flagged['minutes'] == 490 and flagged['starts'] == 6
    assert flagged['code'] == 101 and flagged['id'] == 1
    assert result['paired_player_count'] == 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    # Counter updates and equal metadata instants are neither a news nor status event.
    unchanged = deepcopy(earlier)
    unchanged['elements'][0].update(minutes=490, starts=6,
                                  news_added='2026-10-25T18:00:00+08:00')
    compared = compare(unchanged, earlier)
    assert compared['status_changes'] == compared['news_changes'] == compared['news_metadata_changes'] == []


def test_pair_by_stable_codes_despite_reused_ids_but_never_by_name_or_transfer():
    earlier = snapshot([player(), player(2, 102, status='d'), player(3, 103)])
    current = snapshot([player(91, 101, team=7, status='d'),
                        player(2, 102, team=2, status='i'),
                        player(3, 104, team=7, status='i')])
    # Local team id changed, but stable source team code stayed the same.
    current['teams'][0]['id'] = 7
    result = compare(current, earlier)
    assert result['paired_player_count'] == 1
    assert [r['player_code'] for r in result['status_changes']] == [101]
    assert result['status_changes'][0]['earlier_id'] == 1
    assert result['status_changes'][0]['current_id'] == 91
    assert len(result['unpaired_current_players']) == len(result['unpaired_earlier_players']) == 2
    transfer = next(x for x in result['unpaired_current_players'] if x['player']['code'] == 102)
    assert transfer['reason'] == 'same_player_code_different_team'
    assert transfer['other_snapshot_team_code'] == 10
    # Same name and reused id 3 do not pair code 104 with earlier code 103.
    assert not any(x['player_code'] in (102, 103, 104) for x in result['status_changes'])


def test_unknown_and_other_unavailability_are_not_injury_or_health_claims():
    current = snapshot([player(), player(2, 102, status='u'),
                        player(3, 103, status=None, minutes=None, starts=None),
                        player(4, 104, status='unknown'), player(5, 105, status='s')])
    result = compare(current, current)
    own = result['teams'][0]
    assert own['current_status_counts'] == {'<null_or_missing>': 1, 'a': 1, 's': 1, 'u': 1, 'unknown': 1}
    assert [r['status'] for r in own['flagged_players']] == ['s']
    assert [r['status'] for r in own['unavailable_other_players']] == ['u']
    assert [r['status'] for r in own['unknown_status_players']] == [None, 'unknown']
    assert own['unknown_status_players'][0]['minutes'] is None
    assert result['status_changes'] == result['news_changes'] == []
    assert all('strength_adjustment' not in row and 'starting_probability' not in row
               for row in own['current_players'])


def test_bad_chronology_duplicate_source_keys_and_unknown_team_do_not_guess_pairing():
    valid = snapshot()
    with pytest.raises(ValueError, match='Earlier snapshot'):
        compare_fpl_snapshots(valid, valid, current_observed_at=BEFORE, earlier_observed_at=CURRENT)
    with pytest.raises(ValueError, match='timezone-aware'):
        compare_fpl_snapshots(valid, valid, current_observed_at='2026-10-27T08:00:00', earlier_observed_at=BEFORE)
    bad_inputs = []
    duplicate_team = deepcopy(valid)
    duplicate_team['teams'][1]['code'] = 10
    bad_inputs.append(duplicate_team)
    bad_inputs.append(snapshot([player(), player(2, 101, team=2)]))
    bad_inputs.append(snapshot([player(), player(1, 102)]))
    for bad in bad_inputs:
        with pytest.raises(ValueError, match='Duplicate'):
            compare(bad, valid)
    with pytest.raises(ValueError, match='absent'):
        compare(snapshot([player(team=99)]), valid)

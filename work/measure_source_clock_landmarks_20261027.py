"""Measure two fixed source-clock conventions from saved historical goals.

No per-match offset selection, identity repair, training, or downloads. Goal
minutes are landmarks only: agreeing integers do not verify event sequence.
"""
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import zipfile

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/射门分钟状态重建_20261027'
MANIFEST = ROOT / 'outputs/平局条件信息定位_20261027/EPL分钟资料精确联结明细.json.gz'
SHOTS = ROOT / 'data/understat/earlier_shot_archive_20261027/1099场ID联结射门原字段.json.gz'
MINUTES = ROOT / 'work/research/game_context_20261027/PremierLeague_Sami_minute_by_minute_df_w_red_cards_WITH_BOOKING_ODDS.zip'


def integer(value):
    number = float(value)
    if not number.is_integer():
        raise ValueError(value)
    return int(number)


def side_of(row):
    values = {str(row.get(field)) for field in ('h_a', 'home_away') if row.get(field) is not None}
    if len(values) != 1 or not values <= {'h', 'a'}:
        raise ValueError(('shot side conflict', row['id'], values))
    return next(iter(values))


def counter_rows(counter):
    return [{'credited_side': key[0], 'minute': key[1], 'count': count}
            for key, count in sorted(counter.items())]


def scope_summary(details):
    result = {'matches': len(details), 'with_goals': sum(row['understat_goal_count'] > 0 for row in details),
              'understat_goals': sum(row['understat_goal_count'] for row in details),
              'minute_package_goals': sum(row['minute_package_goal_count'] for row in details),
              'offsets': {}}
    for offset in (0, 1):
        result['offsets'][str(offset)] = {
            'exact_goal_minute_multiset_matches': sum(row['offsets'][str(offset)]['exact'] for row in details),
            'exact_matches_with_goals': sum(row['offsets'][str(offset)]['exact'] and row['understat_goal_count'] > 0 for row in details),
            'matched_goal_landmarks': sum(row['offsets'][str(offset)]['matched'] for row in details),
            'understat_unmatched_goal_landmarks': sum(row['offsets'][str(offset)]['understat_unmatched'] for row in details),
            'minute_package_unmatched_goal_landmarks': sum(row['offsets'][str(offset)]['minute_package_unmatched'] for row in details),
        }
    return result


def main():
    with gzip.open(MANIFEST, 'rt', encoding='utf-8') as stream:
        manifest = json.load(stream)
    with gzip.open(SHOTS, 'rt', encoding='utf-8') as stream:
        shots = json.load(stream)
    targets = {str(row['espn_game_id']): row for row in manifest}
    shot_goals = defaultdict(Counter)
    for row in shots:
        if row['result'] not in ('Goal', 'OwnGoal'):
            continue
        side = side_of(row)
        credited = ('a' if side == 'h' else 'h') if row['result'] == 'OwnGoal' else side
        shot_goals[str(integer(row['match_id']))][(credited, integer(row['minute']))] += 1
    minute_goals = defaultdict(Counter)
    first_half_max = defaultdict(int)
    half_goal_keys = defaultdict(lambda: defaultdict(set))
    with zipfile.ZipFile(MINUTES) as archive:
        members = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected one CSV')
        with archive.open(members[0]) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')):
                espn_id = row['gameId']
                if espn_id not in targets:
                    continue
                half, minute = integer(row['half_id']), integer(row['Minute.clean'])
                if half == 1:
                    first_half_max[espn_id] = max(first_half_max[espn_id], minute)
                count = integer(row['Goals'])
                if count:
                    side = {'Home': 'h', 'Away': 'a'}[row['HomeAway']]
                    minute_goals[espn_id][(side, minute)] += count
                    half_goal_keys[espn_id][(side, minute)].add(half)
    details = []
    for row in manifest:
        espn_id, understat_id = str(row['espn_game_id']), str(row['trusted_understat_id'])
        source = shot_goals[understat_id]
        target = minute_goals[espn_id]
        detail = {key: row[key] for key in ('date', 'season', 'home', 'away', 'espn_game_id', 'trusted_understat_id')}
        detail.update({'minute_goal_accounting_agrees': row['full_time_goal_check'],
                       'understat_goal_count': sum(source.values()),
                       'minute_package_goal_count': sum(target.values()),
                       'author_first_half_maximum_clean_minute': first_half_max[espn_id],
                       'goal_landmarks_on_both_author_halves_at_same_numeric_minute': [
                           {'credited_side': key[0], 'minute': key[1], 'halves': sorted(halves)}
                           for key, halves in half_goal_keys[espn_id].items() if len(halves) > 1],
                       'offsets': {}})
        for offset in (0, 1):
            shifted = Counter({(side, minute + offset): count for (side, minute), count in source.items()})
            matched = sum((shifted & target).values())
            detail['offsets'][str(offset)] = {'exact': shifted == target, 'matched': matched,
                'understat_unmatched': sum((shifted - target).values()),
                'minute_package_unmatched': sum((target - shifted).values())}
            if shifted != target:
                detail['offsets'][str(offset)].update({'shifted_understat': counter_rows(shifted),
                                                      'minute_package': counter_rows(target)})
        details.append(detail)
    accounting_agree = [row for row in details if row['minute_goal_accounting_agrees']]
    summary = {
        'purpose': 'Two fixed global clock hypotheses measured on goal landmarks; no offset is selected per game.',
        'offset_definition': 'Compared minute = Understat minute + offset; ESPN Minute.clean remains unchanged.',
        'all': scope_summary(details), 'minute_goal_accounting_agree': scope_summary(accounting_agree),
        'by_season': {season: scope_summary([row for row in details if row['season'] == season]) for season in sorted({row['season'] for row in details})},
        'minute_goal_accounting_conflicts': [row for row in details if not row['minute_goal_accounting_agrees']],
        'author_first_half_maximum_distribution': dict(sorted(Counter(str(row['author_first_half_maximum_clean_minute']) for row in details).items(), key=lambda item: int(item[0]))),
        'games_with_author_first_half_clean_minute_ge46': sum(row['author_first_half_maximum_clean_minute'] >= 46 for row in details),
        'games_with_goal_landmark_same_numeric_minute_in_both_author_halves': sum(bool(row['goal_landmarks_on_both_author_halves_at_same_numeric_minute']) for row in details),
        'limitations': [
            'Landmark agreement is not independently verified official clock semantics.',
            'Goal-free games contribute no evidence about offset; reported separately.',
            'The minute comparison flattens author half_id and cannot identify a shot period or within-minute event order.',
            'An exact multiset can conceal a period mismatch if both halves share a numerical minute.',
            'No model is fitted, no match identity is changed, and no per-match offset is chosen.',
        ],
        'matches': details,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '时钟地标测量.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('matches', 'minute_goal_accounting_conflicts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

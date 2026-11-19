"""Reconstruct recent EPL opponent/venue expectations using earlier-date fits.

No current forecast, market, LLM probability, or target outcome is loaded.
Only this script's own data directory is written; models use existing presets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import joblib

from football_model.xg_history import DateTeamXgModel
from current_rich_model_helpers_20261026 import load_current_rich_sources
from predict_epl_personnel_context_20261029 import current_independent, instant


ROOT = REPOSITORY_ROOT
SOURCE = ROOT / 'data/epl_personnel_live_20261029/stat_sources'
FACTS = ROOT / 'outputs/英超人员协作前瞻_20261029/给LLM的表现事实包.json'
OUT = ROOT / 'data/epl_adjusted_context_20261029'
CUTOFF = date(2026, 10, 29)


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + '\n', encoding='utf-8')


def load_history():
    current, current_coverage = current_independent()
    sources = load_current_rich_sources(SOURCE / 'EPL_2026.json', current,
                                       before_date=CUTOFF)
    history = sorted((r for r in sources['np_records'].values()
                      if r.competition == 'E0' and r.match_date < CUTOFF),
                     key=lambda r: (r.match_date, r.match_id))
    if len({r.match_id for r in history}) != len(history):
        raise ValueError('Duplicate historical xG identity')
    if max(r.match_date for r in history) != date(2026, 10, 25):
        raise ValueError('Expected the already archived through-October-25 sources')
    return history, {'current_independent_coverage': current_coverage,
                     'rich_source_report': sources['reports']}


def plan(history, facts, *, start=0, count=3, expected_target_date=date(2026, 10, 31)):
    # Original official ordering breaks simultaneous-kickoff ties; no sorting by
    # expected performance or any issued prediction is performed.
    fixtures = facts['records'][start:start+count]
    if len(fixtures) != count:
        raise ValueError('Need the requested original target fixture range')
    all_kickoffs = [instant(f['fixture']['kickoff']) for f in facts['records']]
    if all_kickoffs != sorted(all_kickoffs):
        raise ValueError('Original factual packet is not in official time order')
    by_id = {r.match_id: r for r in history}
    teams = {}
    for fixture in fixtures:
        if expected_target_date is not None and instant(fixture['fixture']['kickoff']).date() != expected_target_date:
            raise ValueError('Target fixture is outside the requested date')
        for side in ('home', 'away'):
            context = fixture['sides'][side]
            team = context['team']
            window = context['window_5']
            original = window['history']
            if team in teams or window['n'] != 5 or len(original) != 5:
                raise ValueError('Need distinct teams with the original five rows')
            rows = [by_id[r['match_id']] for r in original]
            for ref, row in zip(original, rows, strict=True):
                is_home = row.home == team
                if team not in (row.home, row.away):
                    raise ValueError('Original team is absent from referenced match')
                opponent = row.away if is_home else row.home
                if (ref['date'], ref['is_home'], ref['opponent']) != (
                        row.match_date.isoformat(), is_home, opponent):
                    raise ValueError('Original window date/team/side differs from history')
            most_recent = [r.match_id for r in history if team in (r.home, r.away)][-5:]
            if [r.match_id for r in rows] != most_recent:
                raise ValueError('Existing original recent-five window must remain exact')
            teams[team] = rows
    matches = {r.match_id: r for rows in teams.values() for r in rows}
    days = sorted({r.match_date for r in matches.values()})
    return fixtures, teams, matches, days


def describe_plan(history, fixtures, teams, matches, days):
    return {'stage': 'plan', 'history_n': len(history),
            'history_min_date': history[0].match_date.isoformat(),
            'history_max_date': history[-1].match_date.isoformat(),
            'target_fixture_ids': [f['fixture']['fixture_id'] for f in fixtures],
            'team_count': len(teams), 'team_observation_count': sum(map(len, teams.values())),
            'unique_past_match_count': len(matches), 'required_date_model_count': len(days),
            'dates': [d.isoformat() for d in days],
            'prefix_counts': {d.isoformat(): sum(r.match_date < d for r in history) for d in days},
            'parameter_searches': 0, 'historical_prediction_scoring_runs': 0}


def obtain_models(history, days, *, output=OUT, reuse_model_dir=None):
    output = Path(output)
    (output / 'models').mkdir(parents=True, exist_ok=True)
    models, fits = {}, []
    for day in days:
        path = output / 'models' / f'xg_before_{day.isoformat()}.joblib'
        reuse_path = (None if reuse_model_dir is None else
                      Path(reuse_model_dir) / path.name)
        if not path.exists() and reuse_path is not None and reuse_path.exists():
            path = reuse_path
        prefix = [r for r in history if r.match_date < day]
        expected_ids = [r.match_id for r in prefix]
        started = time.perf_counter()
        if path.exists():
            model = joblib.load(path)
            new_fit = False
        else:
            model = DateTeamXgModel(.01, 180).fit(history, day)
            joblib.dump(model, path)
            new_fit = True
        p = model.parameters()
        if (p['reference_match_ids'] != expected_ids
                or p['before_date'] != day.isoformat()
                or p['competition'] != 'E0'
                or p['regularization'] != .01 or p['half_life_days'] != 180):
            raise ValueError('Cached model differs from intended earlier-date fit')
        if p['training_max_date'] >= day.isoformat():
            raise ValueError('Model includes its own fixture date')
        if p['n_matches'] != len(prefix):
            raise ValueError('Historical prefix count differs from fitted count')
        parameters_path = path.with_suffix('.json')
        if new_fit:
            write(parameters_path, p)
        models[day] = model
        fits.append({'before_date': day.isoformat(), 'model_file': str(path),
                     'parameters_file': str(parameters_path), 'new_fit': new_fit,
                     'training_n': len(prefix), 'training_min_date': p['training_min_date'],
                     'training_max_date': p['training_max_date'],
                     'training_match_ids': expected_ids,
                     'optimizer_gradient_max_abs': p['gradient_max_abs'],
                     'elapsed_seconds': time.perf_counter() - started})
        print(json.dumps({'stage': 'past_model_ready',
                          **{k: v for k, v in fits[-1].items() if k != 'training_match_ids'}}),
              flush=True)
    return models, fits


def team_context(team, rows, models, history, *, model_paths=None):
    items = []
    for row in rows:
        model = models[row.match_date]
        p = model.parameters()
        i = 0 if row.home == team else 1
        opponent = (row.away, row.home)[i]
        unknown = sorted({row.home, row.away} - model.teams)
        expected = None if unknown else model.predict(row.home, row.away, 'E0', row.match_date)
        xf, xa = row.xg[i], row.xg[1-i]
        ef, ea = (None, None) if expected is None else (expected[i], expected[1-i])
        attack = None if ef is None else xf - ef
        defense = None if ea is None else xa - ea
        items.append({'match_id': row.match_id, 'understat_id': row.understat_id,
            'date': row.match_date.isoformat(), 'team': team, 'opponent': opponent,
            'is_home': i == 0, 'actual_goals_for': row.goals[i],
            'actual_goals_against': row.goals[1-i],
            'actual_xg_for': xf, 'actual_xg_against': xa,
            'earlier_expected_xg_for': ef, 'earlier_expected_xg_against': ea,
            'attack_xg_residual': attack, 'defense_xga_residual': defense,
            'net_xg_residual': None if attack is None else attack-defense,
            'past_fit_before_date': row.match_date.isoformat(),
            'past_training_n': p['n_matches'], 'past_training_max_date': p['training_max_date'],
            'past_model_file': str((OUT / 'models' / f'xg_before_{row.match_date.isoformat()}.joblib')
                if model_paths is None else model_paths[row.match_date]),
            'own_prior_current_season_matches': sum(r.match_date < row.match_date
                and ':2026/27:' in r.match_id and team in (r.home, r.away) for r in history),
            'opponent_prior_current_season_matches': sum(r.match_date < row.match_date
                and ':2026/27:' in r.match_id and opponent in (r.home, r.away) for r in history),
            'unknown_teams_in_earlier_fit': unknown,
            'missing_reason': None if not unknown else 'unknown_team_in_earlier_fit:' + ','.join(unknown),
            'source_url': row.source_url})
    fields = ('actual_xg_for', 'actual_xg_against', 'earlier_expected_xg_for',
              'earlier_expected_xg_against', 'attack_xg_residual',
              'defense_xga_residual', 'net_xg_residual')
    means = {name: math.fsum(r[name] for r in items)/5
             if len(items) == 5 and all(r[name] is not None for r in items) else None
             for name in fields}
    return {'team': team, 'reference_count': len(rows),
            'original_window_match_ids': [r.match_id for r in rows],
            'available_residual_count': sum(r['net_xg_residual'] is not None for r in items),
            'means_require_all_five': True, 'means': means, 'observations': items}


def check_context(teams, models, history, matches, *, expected_sides=30):
    finite_checked, missing = 0, 0
    for day, model in models.items():
        ids = model.parameters()['reference_match_ids']
        if ids != [r.match_id for r in history if r.match_date < day]:
            raise ValueError('Fitted IDs fail strict earlier-date prefix equality')
        if set(ids) & {r.match_id for r in matches.values() if r.match_date >= day}:
            raise ValueError('A current/later past observation entered a past expectation')
    for context in teams.values():
        if context['reference_count'] != 5:
            raise ValueError('A reference was lost')
        for r in context['observations']:
            if not r['past_training_max_date'] < r['past_fit_before_date'] == r['date'] < CUTOFF.isoformat():
                raise ValueError('Historical expectation date boundary failed')
            if r['net_xg_residual'] is None:
                if not r['unknown_teams_in_earlier_fit']:
                    raise ValueError('Missing model result without an unknown-team reason')
                if any(r[k] is not None for k in ('earlier_expected_xg_for',
                    'earlier_expected_xg_against', 'attack_xg_residual', 'defense_xga_residual')):
                    raise ValueError('Unknown-team expectation was partially invented')
                missing += 1
                continue
            keys = ('actual_xg_for', 'actual_xg_against', 'earlier_expected_xg_for',
                    'earlier_expected_xg_against', 'attack_xg_residual',
                    'defense_xga_residual', 'net_xg_residual')
            if not all(math.isfinite(r[k]) for k in keys):
                raise ValueError('Nonfinite xG/context value')
            if min(r['earlier_expected_xg_for'], r['earlier_expected_xg_against']) <= 0:
                raise ValueError('Nonpositive expected xG')
            for observed, expected in (
                (r['attack_xg_residual'], r['actual_xg_for']-r['earlier_expected_xg_for']),
                (r['defense_xga_residual'], r['actual_xg_against']-r['earlier_expected_xg_against']),
                (r['net_xg_residual'], (r['actual_xg_for']-r['actual_xg_against'])
                    -(r['earlier_expected_xg_for']-r['earlier_expected_xg_against']))):
                if not math.isclose(observed, expected, abs_tol=1e-12):
                    raise ValueError('Residual sign or arithmetic failed')
            finite_checked += 1
        m = context['means']
        if m['net_xg_residual'] is not None:
            if context['available_residual_count'] != 5 or not math.isclose(
                    m['net_xg_residual'], m['attack_xg_residual']-m['defense_xga_residual'], abs_tol=1e-12):
                raise ValueError('Five-match mean arithmetic failed')
        elif context['available_residual_count'] == 5:
            raise ValueError('Complete window lost its mean')
    if finite_checked + missing != expected_sides:
        raise ValueError('Need all original requested side observations')
    return {'date_prefixes_checked': len(models), 'reference_side_count': expected_sides,
            'finite_residual_side_count': finite_checked, 'unknown_residual_side_count': missing,
            'strict_earlier_date_and_training_ids': True,
            'original_five_match_windows_preserved': True,
            'residual_sign_and_mean_arithmetic': True,
            'no_missing_zero_fill': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('inspect', 'run'), default='inspect')
    args = parser.parse_args()
    facts = json.loads(read(FACTS))
    history, source_report = load_history()
    fixtures, team_rows, matches, days = plan(history, facts)
    description = describe_plan(history, fixtures, team_rows, matches, days)
    print(json.dumps(description), flush=True)
    if args.mode == 'inspect':
        return
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / '运行计划.json', description)
    write(OUT / '来源与覆盖.json', source_report)
    write(OUT / '重建输入历史.json', {'cutoff': CUTOFF.isoformat(),
        'competition': 'E0', 'rows': [{'match_id': r.match_id, 'date': r.match_date.isoformat(),
            'home': r.home, 'away': r.away, 'xg': r.xg, 'goals': r.goals,
            'understat_id': r.understat_id, 'source_url': r.source_url} for r in history]})
    models, fits = obtain_models(history, days)
    teams = {team: team_context(team, rows, models, history) for team, rows in team_rows.items()}
    checks = check_context(teams, models, history, matches)
    generated = datetime.now(timezone.utc)
    if any(instant(f['fixture']['kickoff']) <= generated for f in fixtures):
        raise ValueError('A target has started')
    limits = [
        '本文件是模型辅助上下文；以现在存档来源重建过去期待，不声称这些期待在当时已经发布。来源首次发布及历史修订时点不完整。',
        '每场历史期待仅拟合 date < 该历史比赛日期 的英超记录；同日所有比赛均不进入该场期待。不得用今日完整模型回看自身过去比赛。',
        '期待取决于既有 DateTeamXgModel 的结构、正则和时间衰减，主客场和对手攻防由该模型表达；不是独立于该模型的客观真值。',
        '残差表示该球队相对其自身过去对手及主客场条件的较早期待之偏差；不同球队残差不能直接排名为绝对实力。',
        '原始xG、进攻/防守/净残差是同一比赛观测的不同表达，不是新增独立观测；重叠比赛及其双方引用不得重复计权。',
        '原始近5场按原事实包完整保留；未用较早比赛替换缺测，也未按对手或表现挑选。所有均值仅在完整5场时报告。',
        'unknown 队伍保留 null 和可用数量；未对升班球队手写折扣。早期赛季样本少和球队跨季变化仍有不确定性。',
        '未控制红牌、罚球、比分状态、战术、人员、教练变化及完整跨赛事赛程；残差不能宣称完全剔除了赛程影响。',
        'attack residual 为实际xG减过去期待，正值表示创造机会更多；defense residual 为实际xGA减过去期待，正值表示允许机会更多，属防守更差方向。',
        'net residual = attack residual - defense residual。正值仅表示过去净机会表现超过模型期待，不等于本场预期获胜或胜率增量。',
        '未读取市场、当前目标概率、旧LLM概率、目标赛果；未修改原三场判断或十场统计预测；没有新参数搜索、准确率比较或融合。',
    ]
    packet = {'generated_at_utc': generated.isoformat(),
        'scope': {'competition': 'E0', 'cutoff': CUTOFF.isoformat(),
            'fixture_count': 3, 'team_count': 6,
            'fixtures': [f['fixture'] for f in fixtures], 'official_order_preserved': True,
            'source_facts_file': str(FACTS), 'historical_max_date': history[-1].match_date.isoformat(),
            'historical_season_counts': dict(Counter(r.match_id.split(':')[2] for r in history)),
            'side_observation_count': 30, 'unique_past_matches': len(matches)},
        'method': {'estimator': 'existing DateTeamXgModel(.01, 180)',
            'response': 'total match xG (not NPxG or goals)',
            'strict_training_rule': 'historical match date < observation date',
            'conditions': ['league', 'home/away', 'team attack', 'opponent defense'],
            'parameter_searches': 0, 'historical_accuracy_comparisons': 0,
            'mean_weighting': 'equal weight over exactly the five original observations',
            'source_revision_time_reconstructed': False},
        'exposure': {'current_target_probabilities': False, 'market': False,
            'other_llm_probabilities': False, 'target_results': False,
            'historical_model_expected_xg': True},
        'teams': teams, 'fits': fits, 'checks': checks, 'limits': limits}
    write(OUT / '对手条件表现.json', packet)
    raw_expectations = []
    for row in sorted(matches.values(), key=lambda r: (r.match_date, r.match_id)):
        model = models[row.match_date]
        unknown = sorted({row.home, row.away} - model.teams)
        expectation = None if unknown else model.predict(row.home, row.away, 'E0', row.match_date)
        raw_expectations.append({'match_id': row.match_id, 'date': row.match_date.isoformat(),
            'home': row.home, 'away': row.away, 'actual_xg_home_away': row.xg,
            'earlier_expected_xg_home_away': expectation,
            'fit_before_date': row.match_date.isoformat(),
            'training_max_date': model.parameters()['training_max_date'],
            'unknown_teams': unknown, 'understat_id': row.understat_id})
    write(OUT / '逐场过去期待.json', raw_expectations)
    (OUT / '方法与限制.md').write_text(
        '# 英超近5场对手及主客场条件重建\n\n'
        + f'生成于 {generated.isoformat()}，只覆盖官方顺序最早3场、6队、原30侧引用、{len(matches)}场独特历史比赛。\n\n'
        + f'共有 {len(days)} 个日期前缀模型，每个保留 joblib、参数与训练ID/最大日期。\n\n'
        + '\n\n'.join(limits) + '\n', encoding='utf-8')
    print(json.dumps({'stage': 'complete', 'output': str(OUT), 'checks': checks,
        'new_fits': sum(f['new_fit'] for f in fits),
        'means': {t: {'available': c['available_residual_count'], **c['means']}
                  for t, c in teams.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

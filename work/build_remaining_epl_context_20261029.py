"""Extend existing historical conditional xG to the remaining seven fixtures.

Only saved descriptive facts/history and earlier-date model caches are read.
No target probabilities, market, LLM forecast, or target result is consumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import time

from build_epl_adjusted_context_20261029 import (
    ROOT, FACTS, CUTOFF, load_history, plan, describe_plan, obtain_models,
    team_context, check_context, read, write, instant,
)

OUT = ROOT / 'data/epl_remaining_context_20261029'
EARLIER = ROOT / 'data/epl_adjusted_context_20261029'


def verify_original_and_extended_windows(history, facts, fixture_rows, team_rows):
    original_plan = plan(history, facts)
    original = json.loads(read(EARLIER / '对手条件表现.json'))
    if [f['fixture']['fixture_id'] for f in original_plan[0]] != [
            f['fixture_id'] for f in original['scope']['fixtures']]:
        raise ValueError('Default helper scope changed')
    for team, rows in original_plan[1].items():
        if [r.match_id for r in rows] != original['teams'][team]['original_window_match_ids']:
            raise ValueError('Default helper original window changed')
    if set(team_rows) & set(original_plan[1]):
        raise ValueError('Remaining fixtures overlap the original six target teams')
    return {'default_original_three_scope_and_windows_unchanged': True,
            'remaining_fixture_ids_equal_original_records_3_to_9': [
                f['fixture']['fixture_id'] for f in fixture_rows] == [
                f['fixture']['fixture_id'] for f in facts['records'][3:]],
            'side_observations': sum(len(rows) for rows in team_rows.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('inspect', 'run'), default='inspect')
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    facts = json.loads(read(FACTS))
    history, source_report = load_history()
    fixture_rows, team_rows, matches, days = plan(
        history, facts, start=3, count=7, expected_target_date=None)
    window_checks = verify_original_and_extended_windows(history, facts, fixture_rows, team_rows)
    description = describe_plan(history, fixture_rows, team_rows, matches, days)
    cached_days = [day.isoformat() for day in days if (
        EARLIER / 'models' / f'xg_before_{day.isoformat()}.joblib').exists()]
    description.update(original_fixture_record_range=[3, 10],
        existing_earlier_cache_dates=cached_days,
        additional_date_fit_count=len(days)-len(cached_days),
        expected_side_observations=70)
    print(json.dumps(description, ensure_ascii=False), flush=True)
    if args.mode == 'inspect':
        return
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / '运行计划.json', description)
    # Keep this run's coverage but reference the already saved actual xG history.
    write(OUT / '来源与覆盖.json', {
        'source_facts_file': str(FACTS),
        'same_reconstruction_history_file': str(EARLIER / '重建输入历史.json'),
        'same_saved_source_report_file': str(EARLIER / '来源与覆盖.json'),
        'history_n': len(history),
        'historical_source_revision_times_fully_reconstructed': False,
        'current_load_reports': source_report,
    })
    models, fits = obtain_models(history, days, output=OUT,
                                reuse_model_dir=EARLIER / 'models')
    paths = {date.fromisoformat(f['before_date']): f['model_file'] for f in fits}
    contexts = {team: team_context(team, rows, models, history, model_paths=paths)
                for team, rows in team_rows.items()}
    checks = check_context(contexts, models, history, matches, expected_sides=70)
    checks.update(window_checks)
    # Independently re-resolve the selected side and older expectation to catch
    # swapped home/away/opponent fields, not just the residual identity.
    by_id = {r.match_id: r for r in history}
    side_checks = 0
    for team, context in contexts.items():
        for item in context['observations']:
            source = by_id[item['match_id']]
            side = 0 if source.home == team else 1
            if (item['opponent'], item['is_home'], item['actual_xg_for'], item['actual_xg_against']) != (
                    source.away if side == 0 else source.home, side == 0,
                    source.xg[side], source.xg[1-side]):
                raise ValueError('Original source side/opponent/xG mismatch')
            model = models[source.match_date]
            unknown = sorted({source.home, source.away} - model.teams)
            expected = None if unknown else model.predict(
                source.home, source.away, 'E0', source.match_date)
            if expected is None:
                if item['earlier_expected_xg_for'] is not None:
                    raise ValueError('Unknown team acquired an expectation')
            elif (item['earlier_expected_xg_for'], item['earlier_expected_xg_against']) != (
                    expected[side], expected[1-side]):
                raise ValueError('Earlier expected xG side swap')
            side_checks += 1
    checks['source_side_and_expected_index_checks'] = side_checks
    generated = datetime.now(timezone.utc)
    if any(instant(f['fixture']['kickoff']) <= generated for f in fixture_rows):
        raise ValueError('A target has started')
    previous = json.loads(read(EARLIER / '对手条件表现.json'))
    limits = previous['limits'][:-1] + [
        '本轮只扩展原10场中的其余7场；未读取市场、当前目标概率、旧LLM概率或目标赛果，没有修改原统计与LLM预测，没有新参数搜索或准确率比较。',
        '较早13个日期模型按原文件只读复用；只有本范围所需的新日期另按相同设置拟合一次。实际新拟合及缓存复用逐项记录，不把缓存加载称为新训练。',
    ]
    packet = {'generated_at_utc': generated.isoformat(),
        'scope': {'competition': 'E0', 'cutoff': CUTOFF.isoformat(),
            'fixture_count': 7, 'team_count': len(contexts),
            'fixtures': [f['fixture'] for f in fixture_rows],
            'official_order_preserved': True, 'original_fixture_record_range': [3, 10],
            'source_facts_file': str(FACTS),
            'historical_max_date': history[-1].match_date.isoformat(),
            'historical_season_counts': dict(Counter(r.match_id.split(':')[2] for r in history)),
            'side_observation_count': 70, 'unique_past_matches': len(matches)},
        'method': previous['method'], 'exposure': previous['exposure'],
        'teams': contexts, 'fits': fits, 'checks': checks, 'limits': limits,
        'execution': {'started_at_utc': started_at.isoformat(),
            'completed_at_utc': generated.isoformat(),
            'elapsed_seconds': time.perf_counter()-started,
            'new_date_fits': sum(f['new_fit'] for f in fits),
            'cached_date_models_loaded': sum(not f['new_fit'] for f in fits),
            'model_obtain_seconds': sum(f['elapsed_seconds'] for f in fits),
            'parameter_searches': 0, 'target_probabilities_read': False,
            'target_results_read': False}}
    write(OUT / '对手条件表现.json', packet)
    raw_expectations = []
    for row in sorted(matches.values(), key=lambda r: (r.match_date, r.match_id)):
        model = models[row.match_date]
        unknown = sorted({row.home, row.away} - model.teams)
        expected = None if unknown else model.predict(row.home, row.away, 'E0', row.match_date)
        raw_expectations.append({'match_id': row.match_id,
            'date': row.match_date.isoformat(), 'home': row.home, 'away': row.away,
            'actual_xg_home_away': row.xg,
            'earlier_expected_xg_home_away': expected,
            'fit_before_date': row.match_date.isoformat(),
            'training_max_date': model.parameters()['training_max_date'],
            'past_model_file': paths[row.match_date],
            'unknown_teams': unknown, 'understat_id': row.understat_id,
            'source_url': row.source_url})
    write(OUT / '逐场过去期待.json', raw_expectations)
    text = ('# 其余七场英超的历史条件表现\n\n'
        + f'生成于{generated.isoformat()}。原7场、14队、70侧引用，'
        + f'{len(matches)}场独特历史比赛，{len(days)}个日期模型。\n\n'
        + f'新拟合{packet["execution"]["new_date_fits"]}个，缓存加载'
        + f'{packet["execution"]["cached_date_models_loaded"]}个；零参数搜索。'
        + '每个模型及其训练ID/最大日期、原路径均保留。\n\n'
        + '\n\n'.join(limits) + '\n\n'
        + '原窗口、严格日期前缀、训练ID、主客和对手索引、残差符号与均值核对通过。'
        + '本轮没有进行准确率比较或改动目标预测。\n')
    (OUT / '方法与限制.md').write_text(text, encoding='utf-8')
    print(json.dumps({'stage': 'complete', 'output': str(OUT),
        'execution': packet['execution'], 'checks': checks,
        'means': {t: c['means'] for t, c in contexts.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

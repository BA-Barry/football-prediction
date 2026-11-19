"""Finite independent audit of archived historical context; never fit or score."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib


ROOT = REPOSITORY_ROOT
DATA = ROOT / 'data/epl_adjusted_context_20261029'
OUT = ROOT / 'outputs/英超人员协作前瞻_20261029'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def equal(a, b, label):
    if a != b:
        raise AssertionError(f'{label}: {a!r} != {b!r}')


def close(a, b, label):
    if a is None or b is None:
        equal(a, b, label)
    elif not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f'{label}: {a!r} != {b!r}')


def manual_rates(parameters, home, away):
    # Separate hand expression of the two log links, including the defense sign.
    c = parameters['coefficients']
    base = parameters['intercept'] + c.get('competition:E0', 0)
    return (
        math.exp(base + c.get('attack:' + home, 0)
                 - c.get('defense:' + away, 0) + c.get('home:E0', 0)),
        math.exp(base + c.get('attack:' + away, 0)
                 - c.get('defense:' + home, 0)),
    )


def main():
    packet = read(DATA / '对手条件表现.json')
    history = read(DATA / '重建输入历史.json')['rows']
    raw_expectations = read(DATA / '逐场过去期待.json')
    facts = read(OUT / '给LLM的表现事实包.json')
    plan = read(DATA / '运行计划.json')
    by_id = {r['match_id']: r for r in history}
    equal(len(by_id), len(history), 'Unique historical IDs')
    equal(history, sorted(history, key=lambda r: (r['date'], r['match_id'])), 'History order')
    equal(max(r['date'] for r in history), '2026-10-25', 'Archived maximum date')
    equal(all(r['date'] < '2026-10-29' for r in history), True, 'Source cutoff')

    # Project only completed historical source rows. Supplier forecast fields and
    # target results are never accessed or emitted.
    payload = read(ROOT / 'data/epl_personnel_live_20261029/stat_sources/EPL_2026.json')
    source_by_id = {str(r['id']): r for r in payload['dates']
                    if r.get('isResult') is True and r['datetime'][:10] < '2026-10-29'}
    aliases = {'Manchester City': 'Man City', 'Manchester United': 'Man United',
               'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
               'Wolverhampton Wanderers': 'Wolves'}
    source_checked = 0
    for row in history:
        if ':2026/27:' not in row['match_id']:
            continue
        raw = source_by_id[row['understat_id']]
        labels = [aliases.get(raw[s]['title'], raw[s]['title']) for s in ('h', 'a')]
        equal(row['date'], raw['datetime'][:10], 'Source historical date')
        equal([row['home'], row['away']], ['football-data:E0:' + s for s in labels],
              'Source home/away identities')
        for i, side in enumerate(('h', 'a')):
            close(row['xg'][i], float(raw['xG'][side]), 'Archived source total xG')
            equal(row['goals'][i], int(raw['goals'][side]), 'Archived historical score')
        source_checked += 1
    equal(source_checked, len(source_by_id), 'Current source coverage without row loss')

    targets = facts['records'][:3]
    kickoff_times = [datetime.fromisoformat(r['fixture']['kickoff'].replace('Z', '+00:00'))
                     for r in facts['records']]
    equal(kickoff_times, sorted(kickoff_times), 'Original official chronological order')
    equal(packet['scope']['fixtures'], [r['fixture'] for r in targets], 'First three scope/order')
    original = {}
    for fixture in targets:
        for side in ('home', 'away'):
            context = fixture['sides'][side]
            team = context['team']
            equal(team in original, False, 'Distinct target teams')
            original[team] = context['window_5']
    equal(len(original), 6, 'Exactly six teams')
    equal(set(packet['teams']), set(original), 'Team identity set')

    fit_records = {r['before_date']: r for r in packet['fits']}
    models, parameters, fit_audit = {}, {}, []
    required_dates = sorted({r['date'] for w in original.values() for r in w['history']})
    equal(sorted(fit_records), required_dates, 'All and only required date models')
    for day in required_dates:
        fit = fit_records[day]
        model = joblib.load(fit['model_file'])
        p = model.parameters()
        equal(p, read(fit['parameters_file']), 'Saved joblib/parameter JSON equality')
        expected = [r for r in history if r['date'] < day]
        expected_ids = [r['match_id'] for r in expected]
        equal(p['reference_match_ids'], expected_ids, 'Model exact earlier prefix IDs')
        equal(fit['training_match_ids'], expected_ids, 'Packet exact earlier prefix IDs')
        equal(p['before_date'], day, 'Fit cutoff')
        equal(p['training_max_date'], max(r['date'] for r in expected), 'Training maximum date')
        equal(p['training_min_date'], min(r['date'] for r in expected), 'Training minimum date')
        equal(p['n_matches'], len(expected), 'Model training size')
        equal(fit['training_n'], len(expected), 'Packet training size')
        equal(plan['prefix_counts'][day], len(expected), 'Plan training size')
        equal(p['training_max_date'] < day, True, 'Exclude entire own date')
        equal(p['competition'], 'E0', 'Model league')
        equal((p['regularization'], p['half_life_days']), (.01, 180), 'Existing I1 preset')
        equal(set(p['teams']), {s for r in expected for s in (r['home'], r['away'])},
              'Known teams equal earlier training union')
        equal(set(p['teams']), model.teams, 'Parameter and joblib team set')
        fit_audit.append({'before_date': day, 'training_n': len(expected),
                          'training_max_date': p['training_max_date'],
                          'own_or_later_date_training_id_count':
                          sum(by_id[i]['date'] >= day for i in p['reference_match_ids'])})
        models[day], parameters[day] = model, p

    raw_by_id = {r['match_id']: r for r in raw_expectations}
    equal(len(raw_by_id), len(raw_expectations), 'Unique raw expectation rows')
    rows_audit, teams_audit, unique_past = [], [], set()
    fields = ('actual_xg_for', 'actual_xg_against', 'earlier_expected_xg_for',
              'earlier_expected_xg_against', 'attack_xg_residual',
              'defense_xga_residual', 'net_xg_residual')
    max_error = 0.0
    for team, window in original.items():
        context = packet['teams'][team]
        equal((window['n'], window['requested_n'], len(window['history'])), (5, 5, 5),
              'Original complete five')
        original_ids = [r['match_id'] for r in window['history']]
        equal(context['original_window_match_ids'], original_ids, 'Original IDs preserved')
        equal([r['match_id'] for r in context['observations']], original_ids, 'All original rows')
        equal(original_ids, [r['match_id'] for r in history if team in (r['home'], r['away'])][-5:],
              'No chosen available/older substitution')
        equal(context['reference_count'], 5, 'Reference count')
        equal(context['means_require_all_five'], True, 'No partial mean')
        available = 0
        for ref, observed in zip(window['history'], context['observations'], strict=True):
            row = by_id[ref['match_id']]
            unique_past.add(row['match_id'])
            equal(team in (row['home'], row['away']), True, 'Team present')
            own = 0 if team == row['home'] else 1
            opponent = row['away'] if own == 0 else row['home']
            equal((observed['team'], observed['opponent'], observed['is_home'], observed['date']),
                  (team, opponent, own == 0, row['date']), 'Observation side identities')
            equal((ref['opponent'], ref['is_home'], ref['date']),
                  (opponent, own == 0, row['date']), 'Original side identities')
            close(observed['actual_xg_for'], row['xg'][own], 'Own source xG index')
            close(observed['actual_xg_against'], row['xg'][1-own], 'Opponent source xG index')
            equal((observed['actual_goals_for'], observed['actual_goals_against']),
                  (row['goals'][own], row['goals'][1-own]), 'Historical goals side index')
            p = parameters[row['date']]
            unknown = sorted({row['home'], row['away']} - set(p['teams']))
            equal(observed['unknown_teams_in_earlier_fit'], unknown, 'Unknown teams exact')
            equal(observed['past_fit_before_date'], row['date'], 'Observation cutoff')
            equal(observed['past_training_n'], p['n_matches'], 'Observation training size')
            equal(observed['past_training_max_date'], p['training_max_date'], 'Observation prefix max')
            equal(row['match_id'] not in p['reference_match_ids'], True, 'No own-match training ID')
            expected_values = dict.fromkeys(fields[2:], None)
            if unknown:
                equal(observed['missing_reason'], 'unknown_team_in_earlier_fit:' + ','.join(unknown),
                      'Unknown reason')
            else:
                equal(observed['missing_reason'], None, 'Known expectation reason')
                rates = manual_rates(p, row['home'], row['away'])
                from_joblib = models[row['date']].predict(row['home'], row['away'], 'E0',
                                                         datetime.fromisoformat(row['date']).date())
                for a, b in zip(rates, from_joblib, strict=True):
                    close(a, b, 'Manual log link equals archived model')
                ef, ea = rates[own], rates[1-own]
                xf, xa = row['xg'][own], row['xg'][1-own]
                expected_values = dict(zip(fields[2:], (ef, ea, xf-ef, xa-ea,
                                                        (xf-xa)-(ef-ea)), strict=True))
                available += 1
            for key, value in expected_values.items():
                close(observed[key], value, 'Independent observation ' + key)
                if value is not None:
                    max_error = max(max_error, abs(observed[key]-value))
            raw = raw_by_id[row['match_id']]
            equal((raw['home'], raw['away'], raw['date']), (row['home'], row['away'], row['date']),
                  'Raw expectation identity')
            equal(raw['unknown_teams'], unknown, 'Raw unknown teams')
            if unknown:
                equal(raw['earlier_expected_xg_home_away'], None, 'Unknown raw rates not zero')
            else:
                for a, b in zip(raw['earlier_expected_xg_home_away'], rates, strict=True):
                    close(a, b, 'Raw home/away expectation')
            rows_audit.append({'team': team, 'match_id': row['match_id'], 'date': row['date'],
                'is_home': own == 0, 'opponent': opponent, 'unknown_teams': unknown,
                'earlier_expected_xg_for': observed['earlier_expected_xg_for'],
                'earlier_expected_xg_against': observed['earlier_expected_xg_against'],
                'attack_xg_residual': observed['attack_xg_residual'],
                'defense_xga_residual': observed['defense_xga_residual'],
                'net_xg_residual': observed['net_xg_residual']})
        equal(context['available_residual_count'], available, 'Complete known count')
        for key in fields:
            values = [r[key] for r in context['observations']]
            expected = sum(values)/5 if all(v is not None for v in values) else None
            close(context['means'][key], expected, 'Independent exactly-five mean ' + key)
        close(context['means']['actual_xg_for'], window['measurements']['xg_for'], 'Original raw xG mean')
        close(context['means']['actual_xg_against'], window['measurements']['xg_against'], 'Original raw xGA mean')
        teams_audit.append({'team': team, 'original_reference_count': 5,
                            'available_residual_count': available, 'means': context['means']})

    equal(len(rows_audit), 30, 'All thirty side observations')
    equal(set(raw_by_id), unique_past, 'Raw expectation set exactly observed matches')
    equal(len(unique_past), packet['scope']['unique_past_matches'], 'Unique match count')
    equal(packet['method']['historical_accuracy_comparisons'], 0, 'No accuracy claim')
    equal(packet['method']['parameter_searches'], 0, 'No parameter search')
    equal(packet['method']['source_revision_time_reconstructed'], False, 'Publication time limitation')
    equal(packet['exposure'], {'current_target_probabilities': False, 'market': False,
          'other_llm_probabilities': False, 'target_results': False,
          'historical_model_expected_xg': True}, 'Declared exposure boundary')
    text = '\n'.join(packet['limits'])
    semantic = {
        'reconstructed_now_not_published_then': '不声称这些期待在当时已经发布' in text,
        'source_first_publication_and_revision_unknown': '首次发布' in text and '修订' in text,
        'not_complete_injury_card_tactics_control': all(s in text for s in ('红牌', '战术', '人员')),
        'positive_defense_is_worse': '属防守更差方向' in text,
        'net_not_win_probability_increment': '不等于本场预期获胜或胜率增量' in text,
        'no_accuracy_comparison': '没有新参数搜索、准确率比较或融合' in text,
        'own_baseline_not_absolute_cross_team_strength':
            ('自己' in text or '自身' in text) and '绝对实力' in text,
        'same_xg_observation_not_independent_evidence':
            '独立观测' in text and ('同一' in text or '不同表达' in text),
    }
    # This review reports findings; it does not impose a new production gate.
    findings = [key for key, passed in semantic.items() if not passed]
    summary = {'status': 'passed' if not findings else 'numerical_pass_with_explanation_findings',
        'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'one finite independent source/identity/date/model/arithmetic/explanation audit',
        'input_packet_generated_at_utc': packet['generated_at_utc'],
        'history_count': len(history), 'current_season_source_rows_crosschecked': source_checked,
        'team_count': 6, 'side_count': len(rows_audit), 'unique_past_matches': len(unique_past),
        'date_model_count': len(required_dates),
        'known_residual_side_count': sum(not r['unknown_teams'] for r in rows_audit),
        'unknown_residual_side_count': sum(bool(r['unknown_teams']) for r in rows_audit),
        'maximum_manual_reconstruction_absolute_error': max_error,
        'strict_earlier_date_and_exact_training_ids': True,
        'original_window5_all_thirty_side_rows_retained': True,
        'home_away_and_opponent_indices_checked': True,
        'actual_minus_expected_attack_minus_defense_and_means_checked': True,
        'unknowns_not_zero_or_selected_out': True,
        'semantic_checks': semantic, 'findings': findings,
        'review_execution': {'model_fits': 0, 'network_requests': 0, 'parameter_searches': 0,
            'accuracy_comparisons': 0, 'target_probabilities_accessed': False,
            'target_outcomes_accessed': False},
        'fit_audit': fit_audit, 'team_audit': teams_audit, 'side_audit': rows_audit,
        'interpretation': [
            '残差描述球队相对自己在历史对手和主客条件下较早模型基准的偏差；不同球队残差不构成可直接比较的绝对实力尺度。',
            '原xG与进攻、防守、净残差均由同一观测导出，不能按新增独立观测重复计权。',
            '历史期待由现在存档数据重建，并非当时已发布；来源首次发布和修订时点未知。',
            '未完全控制伤停、红牌、战术、比分状态和人员变化；此审查不评估是否提高预测准确率。',
        ]}
    (OUT / '对手条件接入独立复核.json').write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 对手条件接入独立复核', '',
        f'结论：{summary["status"]}。完成一次有限独立核对，之后停止。', '',
        f'核对原6队近5场全部30侧、{len(unique_past)}场独特旧场、{len(required_dates)}个历史日期模型；'
        f'80场本季原xG/身份与存档来源交叉核对。{summary["known_residual_side_count"]}侧残差可用，'
        f'{summary["unknown_residual_side_count"]}侧保留未知。', '',
        '每个模型训练ID与严格 date < 旧场日期 的历史前缀逐项一致，同日或未来训练ID为0。'
        '保留原window5顺序与全部30侧；未填0、未选取可用样本或替换旧场。', '',
        f'独立按保存系数手算主客期待，再核对actual−expected、attack−defense及恰好五场均值，'
        f'最大绝对数值差为{max_error:.3g}。对手身份与主客索引一致。', '',
        *summary['interpretation'], '',
        '本复核新增训练0、联网0、参数搜索0、准确率比较0；未读取当前预测概率或目标赛果。', '',
        ('解释层检查全部通过。' if not findings else '解释层尚需补充：' + '；'.join(findings)), '',
        '|球队|原窗口|残差可用|进攻残差/场|防守残差/场（正=更差）|净残差/场|',
        '|---|---:|---:|---:|---:|---:|']
    for t in teams_audit:
        fmt = lambda key: '未知' if t['means'][key] is None else f'{t["means"][key]:+.6f}'
        lines.append('|'+t['team'].split(':')[-1]+'|5|'+str(t['available_residual_count'])+'|'
            +'|'.join(fmt(k) for k in ('attack_xg_residual', 'defense_xga_residual', 'net_xg_residual'))+'|')
    (OUT / '对手条件接入独立复核.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('status', 'history_count', 'team_count', 'side_count',
        'unique_past_matches', 'date_model_count', 'known_residual_side_count',
        'unknown_residual_side_count', 'maximum_manual_reconstruction_absolute_error', 'findings')},
        ensure_ascii=True))


if __name__ == '__main__':
    main()

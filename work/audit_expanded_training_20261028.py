"""Read-only reconstruction of the existing expansion artifacts; never fit."""
from collections import Counter
from datetime import date, datetime, timezone, timedelta
import gzip
import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import skellam

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.fixed_horizon_features import PreparedPredictionFeatures
from run_broad_early_models_20261025 import observations
from run_npxg_comparison_20261025 import load_source

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练'
FEATURE = 'fresh_performance_features'


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    old = read(ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/features_lead3.json.gz')['rows']
    merged = read(OUT/'扩样合并特征.json.gz')['rows']
    added_file = read(OUT/'2021-22新增特征.json.gz')
    added = added_file['rows']
    original_by_id = {r['match_id']: r for r in old}
    merged_by_id = {r['match_id']: r for r in merged}
    assert len(old) == len(original_by_id) == 6967
    assert len(merged) == len(merged_by_id) == 8753
    assert len(added) == 1786
    assert all(merged_by_id[ident] == row for ident, row in original_by_id.items())
    assert all(merged_by_id[r['match_id']] == r for r in added)
    assert not set(original_by_id).intersection(r['match_id'] for r in added)
    source, independent, *_ = load_source()
    original_model = joblib.load(ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z/丰富信息候选模型.joblib')
    original_parameters = original_model.parameters()
    names = set(original_parameters['feature_names'])
    assert len(names) == 121
    parameter_records = read(OUT/'新增基础模型参数.json')
    assert len(parameter_records) == 55
    for key, pair in parameter_records.items():
        comp, cutoff = key.split(':')
        expected = sorted((r for mid, r in source.items() if r.competition == comp
            and r.match_date.isoformat() < cutoff and independent[mid].goal_fit_exclusion is None),
            key=lambda r: (r.match_date, r.match_id))
        expected_ids = [r.match_id for r in expected]
        for arm in ('goal', 'xg'):
            actual = pair[arm]
            assert actual['before_date'] == cutoff
            assert actual['reference_match_ids'] == expected_ids
            assert len(expected_ids) == actual['n_matches']
            assert max(independent[mid].match_date.isoformat() for mid in expected_ids) < cutoff
            assert min(independent[mid].match_date.isoformat() for mid in expected_ids) >= '2020-08-01'

    builder = PreparedPredictionFeatures(observations(source), AllCompetitionFeatures([]))
    cached_pairs = {}
    max_rate_error = 0.
    for row in added:
        played = date.fromisoformat(row['date'])
        forecast = played-timedelta(days=3)
        anchor = date(2021, 8, 1)
        parameter = anchor+timedelta(days=28*((forecast-anchor).days//28))
        assert row['fit_before_date'] == parameter.isoformat()
        assert row['forecast_before_date'] == forecast.isoformat()
        assert row['fresh_feature_before_date'] == forecast.isoformat()
        assert parameter <= forecast < played
        assert row['period'] == '2021-22'
        ident = row['match_id']
        actual = independent[ident]
        assert row['actual_score'] == list(actual.full_time)
        assert row['actual_outcome'] == (0 if actual.full_time[0] > actual.full_time[1]
                                        else 2 if actual.full_time[0] < actual.full_time[1] else 1)
        key = row['base_parameter_key']
        for arm in ('goal', 'xg'):
            assert ident not in parameter_records[key][arm]['reference_match_ids']
        for side in row['fresh_feature_history']['history'].values():
            assert all(independent[mid].match_date < forecast for mid in side['used_match_ids'])
            assert ident not in side['used_match_ids']
        for side in row[FEATURE]:
            assert set(side) == names
        if key not in cached_pairs:
            comp, cutoff = key.split(':')
            cached_pairs[key] = joblib.load(OUT/'基础模型'/f'{comp}_{cutoff}.joblib')
        pair = cached_pairs[key]
        rates = {arm: pair[arm].predict(row['home'], row['away'], row['competition'], played)
                 for arm in ('goal', 'xg')}
        for arm in rates:
            max_rate_error = max(max_rate_error, max(abs(a-b) for a, b in zip(rates[arm], row[f'{arm}_rates'])))
        rebuilt = builder.for_match(home=row['home'], away=row['away'], competition=row['competition'],
            parameter_cutoff=parameter, forecast_cutoff=forecast, match_date=played,
            goal_rates=rates['goal'], xg_rates=rates['xg'], refresh=True)
        assert rebuilt['performance_features'] == row[FEATURE]
        assert rebuilt['feature_history'] == row['fresh_feature_history']

    unavailable = added_file['unavailable']
    assert len(unavailable) == 36
    assert len(added)+len(unavailable) == added_file['target_source_n'] == 1822
    for row in unavailable:
        key = f"{row['competition']}:{row['fit_before_date']}"
        actual_missing = {arm: sorted({row['home'], row['away']}-set(parameter_records[key][arm]['teams']))
                          for arm in ('goal', 'xg')}
        assert actual_missing == row['unseen_teams']
        assert any(actual_missing.values())
    held = sorted((r for r in old if r['period'] == '2025-26'), key=lambda r:(r['date'], r['match_id']))
    held_ids = {r['match_id'] for r in held}
    summary = read(OUT/'扩样训练结果.json')
    checked_models, scoring = {}, {}
    for name, cohort, cutoff in [('原覆盖_20250801', old, date(2025,8,1)),
                                  ('扩样_20250801', merged, date(2025,8,1)),
                                  ('扩样_20260801_研究候选', merged, date(2026,8,1))]:
        model = joblib.load(OUT/f'{name}.joblib')
        parameters = model.parameters()
        training = sorted((r for r in cohort if r['date'] < cutoff.isoformat()), key=lambda r:(r['date'],r['match_id']))
        expected_ids = [r['match_id'] for r in training]
        assert expected_ids == parameters['reference_match_ids']
        assert parameters['before_date'] == cutoff.isoformat()
        assert parameters['configuration'] == original_parameters['configuration']
        assert set(parameters['feature_names']) == names
        overlap = held_ids.intersection(expected_ids)
        if cutoff.year == 2025:
            assert not overlap
            saved = read(OUT/f'{name}_2025-26预测.json')
            assert [r['match_id'] for r in held] == [r['match_id'] for r in saved]
            predicted = model.predict_many([r[FEATURE] for r in held], [date.fromisoformat(r['date']) for r in held])
            archived_rates = np.asarray([r['rates'] for r in saved])
            rate_error = float(np.max(np.abs(np.asarray(predicted)-archived_rates)))
            assert rate_error <= 1e-12
            p = np.asarray([r['probabilities'] for r in saved])
            y = np.asarray([r['actual_outcome'] for r in saved])
            assert (p >= 0).all() and np.max(abs(p.sum(axis=1)-1)) <= 1e-12
            assert all(r['actual_score'] == h['actual_score'] and r['actual_outcome'] == h['actual_outcome'] for r,h in zip(saved,held))
            ll = float(-np.log(p[np.arange(len(y)),y]).mean())
            brier = float(((p-np.eye(3)[y])**2).sum(axis=1).mean())
            correct = int((p.argmax(axis=1)==y).sum())
            reported = summary['scores'][name]['one_x_two']
            assert abs(ll-reported['log_loss']) <= 1e-12
            assert abs(brier-reported['brier_sum']) <= 1e-12
            assert abs(correct/len(y)-reported['accuracy']) <= 1e-12
            # Independent untruncated Poisson difference calculation; archived
            # grid uses finite, negligible-tail normalization.
            mathematical_p = np.stack([skellam.sf(0,archived_rates[:,0],archived_rates[:,1]),
                skellam.pmf(0,archived_rates[:,0],archived_rates[:,1]),
                skellam.cdf(-1,archived_rates[:,0],archived_rates[:,1])],axis=1)
            law_error = float(np.max(abs(mathematical_p-p)))
            assert law_error < 1e-8
            scoring[name] = {'n':len(y), 'correct':correct, 'accuracy':correct/len(y),
                'log_loss':ll, 'brier_sum':brier, 'saved_model_rate_replay_max_error':rate_error,
                'independent_skellam_probability_max_difference':law_error}
        else:
            assert overlap == held_ids
        weights = np.asarray([2**(-(cutoff-date.fromisoformat(r['date'])).days/180) for r in training])
        ess = float(weights.sum()**2/(weights@weights))
        checked_models[name] = {'training_n':len(training), 'cutoff':cutoff.isoformat(),
            'evaluation_ids_in_training_n':len(overlap),
            'used_for_2025_26_evaluation':cutoff.year==2025,
            'weight_concentration_kish_n':ess,
            'interpretation':'Weight concentration; neither independent observation count nor evidence of calibration.'}
        if cutoff.year == 2026:
            assert abs(ess-summary['final_weight_summary']['effective_match_n_kish']) < 1e-9

    result = {'audited_at_utc':datetime.now(timezone.utc).isoformat(), 'new_fits':0,
        'network_calls':0, 'new_outcome_queries':0,
        'old_rows_unchanged_n':len(old), 'expanded_unique_n':len(merged),
        'added_full_feature_reconstructions_equal_n':len(added),
        'added_rate_replay_max_error':max_rate_error,
        'base_parameter_pairs_checked':len(parameter_records),
        'unavailable_reasons_confirmed_n':len(unavailable),
        'models':checked_models,'held_predictions':scoring,
        'findings':[],
        'limitations':['The final 2026 candidate includes all 1726 old evaluation matches, as legitimately earlier training, and was not used to score them.',
            'Historical seasons were already examined for development; this is not fresh blind or present-day accuracy.',
            'Calendar-date exclusion does not reconstruct historical first-publication time or vendor revisions.',
            'Upstream experts refresh during the evaluation season using earlier matches; this is not a fully static upstream experiment.',
            'No claim of independent effective observations is made from Kish concentration.',
            'This audit reads saved sources and reproduces calculations; it does not authenticate vendor history.']}
    (OUT/'独立核对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines=['# 扩样训练独立核对','',f"核对时刻：{result['audited_at_utc']}。本轮新拟合0、网络请求0、赛果查询0。",'',
        '- 原6,967行逐对象保持；合并8,753个唯一比赛ID。',
        '- 新增1,786行全部从已保存基础模型和较早来源重建，121项特征及历史元数据逐字段一致；55组基础训练ID均严格早于参数日，目标自身未进入。',
        '- 36个不可用目标均确认属于当时基础模型未见球队；没有硬填零或按赛果删选。',
        '- 用于2025/26评分的两棵树训练ID均与1,726个评分目标完全分离；全1,726场从保存模型重放率误差不超过1e-12。',
        '- 最终2026研究候选合法使用了这1,726场过去标签；它没有用于报告该批持出成绩，不能将其训练内结果说成验证。','',
        '|固定截止2025-08-01的训练|同一1726场正确数|LogLoss|Brier|',
        '|---|---:|---:|---:|']
    for name,r in scoring.items():
        lines.append(f"|{name}|{r['correct']}|{r['log_loss']:.9f}|{r['brier_sum']:.9f}|")
    lines.extend(['','上述评分独立重算与原报告一致；另用无截尾Skellam分布检查完整H/D/A，最大差见JSON。',
        '', '局限：已查看的研发赛季不是新盲测；基础专家在评分季按较早日期更新。180天衰减的Kish值只描述权重集中度，不是球队比赛之间的独立样本量。首次发布时间、源修订与供应商历史真实性不在本次验证能力内。'])
    (OUT/'独立核对.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()

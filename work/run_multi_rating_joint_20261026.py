"""Run two complete, fixed multi-rating XGBoost candidates on saved D-3 rows.

Reuses real archived observations and old forecasts. No import-time training,
network request, current-result check, score reconciliation or model selection.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.evaluation import probability_scores, calibration_bins
from football_model.forecast_diagnostics import moving_block_means
from football_model.multi_rating import build_rating_features
from football_model.multi_rating_xgb import FixedMultiRatingXGB, FIXED_CONFIGURATION


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/多路评级联合接入_20261026'
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'
AVAILABILITY = ROOT/'outputs/FPL历史可用性接入_20261026/逐场赛前状态特征.json.gz'
EXPOSURE = ROOT/'outputs/FPL可用性变化与参与接入_20261026/逐场变化参与特征.json.gz'
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
PERIODS = ('2024-25', '2025-26')
ARMS = ('rating_core', 'rich_joint')
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away',
           'actual_score', 'actual_outcome', 'fit_before_date',
           'forecast_before_date', 'horizon_days')


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def prepare():
    original = read(SOURCE/'features_lead3.json.gz')['rows']
    original.sort(key=lambda r: (r['date'], r['match_id']))
    avail = {r['match_id']: r for r in read(AVAILABILITY)['rows']}
    exposure = {r['match_id']: r for r in read(EXPOSURE)['rows']}
    anames = sorted(set(next(iter(avail.values()))['side_features'][0]) -
                    set(next(iter(avail.values()))['reference_features'][0]))
    enames = sorted(set(next(iter(exposure.values()))['side_features'][0]) -
                    set(next(iter(exposure.values()))['reference_features'][0]))
    observations = read(OUT/'历史评级观测.json')['observations']
    targets = [{'match_id': r['match_id'], 'forecast_date': r['forecast_before_date'],
                'league': r['competition'], 'home_team': r['home'], 'away_team': r['away']}
               for r in original]
    bank = build_rating_features(observations, targets)
    ratings = {r['match_id']: r for r in bank['rows']}
    write(OUT/'评级生成说明.json', bank['metadata'])
    rows = []
    for r in original:
        if not r['fit_before_date'] <= r['forecast_before_date'] < r['date']:
            raise ValueError('Retain the original parameter/publication/fixture ordering')
        h, a = r['actual_score']
        if r['actual_outcome'] != (0 if h > a else 1 if h == a else 2):
            raise ValueError('Local H/D/A labels must agree with full scores')
        raw = r['fresh_allcomp_features'][0]
        core = {**ratings[r['match_id']]['features'],
                'calendar_day_of_year': float(date.fromisoformat(r['date']).timetuple().tm_yday),
                'home_league_gap_to_match': raw.get('own_same_league_gap_to_match_days'),
                'away_league_gap_to_match': raw.get('opponent_same_league_gap_to_match_days'),
                **{f'league_{c}': float(r['competition'] == c) for c in COMPS}}
        joint = {**core, **raw, 'availability_source_scope_epl': float(r['competition'] == 'E0')}
        for tag, source, names in (('status', avail, anames), ('usage_status', exposure, enames)):
            item = source.get(r['match_id'])
            if item is not None:
                if any(item[k] != r[k] for k in CONTEXT):
                    raise ValueError('Exact identity/date/score match needed for archived input reuse')
                expected = r['fresh_performance_features'][0]
                if item['reference_features'][0] != expected:
                    raise ValueError('Player descriptors must refer to the identical original history')
            for name in names:
                joint[f'av_{tag}_{name}'] = None if item is None else item['side_features'][0].get(name)
        rows.append({**{k: r[k] for k in CONTEXT}, 'feature_before_date': r['forecast_before_date'],
                     'rating_core_features': core, 'rich_joint_features': joint,
                     'rating_history_max_date': ratings[r['match_id']]['history_max_date']})
    names = {arm: sorted(rows[0][arm+'_features']) for arm in ARMS}
    with gzip.open(OUT/'联合赛前输入.json.gz', 'wt', encoding='utf-8') as f:
        json.dump({'rows': rows}, f, ensure_ascii=False, allow_nan=False)
    coverage = {'n_rows': len(rows), 'by_period': dict(Counter(r['period'] for r in rows)),
        'feature_counts': {k: len(v) for k,v in names.items()}, 'feature_names': names,
        'availability_extra_measurements': {'raw_status': anames, 'usage_status': enames},
        'by_league': {}, 'rating_metadata': bank['metadata'],
        'source_inputs': [str(SOURCE/'features_lead3.json.gz'), str(AVAILABILITY), str(EXPOSURE)],
        'timing': 'date-level only: observations < publication day; fit outcomes < parameter day',
        'unknown': 'non-EPL availability is None, never inferred healthy; native missing routing',
        'adaptation': 'official equations and fixed learner skeleton plus locally dated full information; not paper reproduction'}
    for c in COMPS:
        local = [r for r in rows if r['competition'] == c]
        coverage['by_league'][c] = {'n': len(local), 'missing_by_joint_feature': {
            n: sum(r['rich_joint_features'].get(n) is None for r in local) for n in names['rich_joint']}}
    write(OUT/'联合接入实测.json', coverage)
    print(json.dumps({'prepared_rows': len(rows), 'feature_counts': coverage['feature_counts'],
                      'shot_adjustment_fits': bank['metadata']['new_shot_adjustment_fits']}, ensure_ascii=False), flush=True)
    return rows, coverage


def score(rows):
    p = np.asarray([r['probabilities'] for r in rows], dtype=float)
    y = np.asarray([r['actual_outcome'] for r in rows], dtype=int)
    if np.any(p <= 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-10):
        raise ValueError('Use finite positive normalized model probabilities without score clipping')
    pred = p.argmax(axis=1)
    cm = np.zeros((3,3), dtype=int)
    np.add.at(cm, (y,pred), 1)
    bins = calibration_bins(p,y,bins=10)
    return probability_scores(p,y) | {
        'correct': int((pred == y).sum()), 'confusion_actual_rows_predicted_columns': cm.tolist(),
        'class_recall': {name: float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None
                         for i,name in enumerate(('home','draw','away'))},
        'predicted_counts': dict(zip(('home','draw','away'), np.bincount(pred,minlength=3).tolist())),
        'mean_probabilities': p.mean(axis=0).tolist(),
        'observed_frequencies': (np.bincount(y,minlength=3)/len(y)).tolist(),
        'calibration_bins': bins,
        'class_mean_ece_10_equal_width': float(sum(b['n']*abs(b['mean_probability']-b['observed_frequency'])
                                                 for b in bins)/(3*len(y))),
        'mean_top_probability': float(p.max(axis=1).mean())}, {
        'log_loss': -np.log(p[np.arange(len(y)),y]),
        'brier_sum': np.square(p-np.eye(3)[y]).sum(axis=1),
        'error': (pred != y).astype(float)}


def comparison(tracks, intervals=False):
    ref = tracks['saved_performance']
    ids = [r['match_id'] for r in ref]
    scores, losses = {}, {}
    for arm, values in tracks.items():
        if ids != [r['match_id'] for r in values]:
            raise ValueError('Retain the complete identical paired cohort for every route')
        if any(any(a[k] != b[k] for k in CONTEXT) for a,b in zip(ref,values)):
            raise ValueError('Matched predictions must describe identical contexts')
        scores[arm], losses[arm] = score(values)
    columns, deltas = [], []
    for new, old in [('rating_core','saved_performance'),('rich_joint','saved_performance'),
                     ('rich_joint','saved_allcomp'),('rich_joint','rating_core')]:
        for metric in ('log_loss','brier_sum','error'):
            columns.append(f'{new}_minus_{old}:{metric}')
            deltas.append(losses[new][metric]-losses[old][metric])
    matrix = np.column_stack(deltas)
    result = {'n': len(ids), 'scores': scores, 'difference_order': columns,
              'mean_differences': dict(zip(columns,matrix.mean(axis=0).tolist()))}
    if intervals:
        result['conditional_four_week_intervals'] = moving_block_means(
            [r['date'] for r in ref],matrix,block_weeks=4)
    return result


def run(rows, coverage):
    model_dir = OUT/'models'
    model_dir.mkdir(exist_ok=True)
    method = {'created_at': datetime.now(timezone.utc).isoformat(),
        'source_repository': 'https://github.com/spoho-datascience/sparsity-sports-forecasting',
        'candidates': list(ARMS), 'classifier_configuration': FIXED_CONFIGURATION,
        'weighting': 'uniform, following upstream; not a causal algorithm-only comparison with time-weighted regression',
        'historical_scope': 'already repeatedly examined 2024/25 and 2025/26 seasons; chronological research, not untouched test',
        'fit_rule': 'reuse original 28-day fit dates and D-3 feature snapshots; no same-day outcomes',
        'missing': 'native XGBoost NaN; non-EPL availability unknown; all original target matches retained',
        'primary_use': '1X2 accuracy and probability quality together; no requirement that every member win alone',
        'other_reports': ['Brier','RPS','draw recall','class confusion','calibration','per-league support'],
        'stop': 'two fixed whole candidates x 22 original dates, then stop without rescue tuning or selecting fields on later scores',
        'no_new_parameter_grid': True, 'no_early_stopping': True,
        'no_neural_network': True, 'no_market_quote_or_target_XI': True,
        'score_scope': 'direct 1X2 only; original score model remains separate, no implicit reconciliation',
        'current_forecasts_replaced': False, 'current_results_checked': False,
        'adaptations': coverage['rating_metadata']['source_parameter_differences']}
    write(OUT/'本轮方法.json', method)
    old = read(SOURCE/'lead_3/逐场完整预测.json')
    tracks = {arm: [] for arm in ARMS}
    for name,source in [('saved_performance','fresh_performance'),('saved_allcomp','fresh_allcomp')]:
        tracks[name] = [{**{k:r[k] for k in CONTEXT},'probabilities':r['probabilities']} for r in old[source]]
        tracks[name].sort(key=lambda r:(r['date'],r['match_id']))
    windows, params = [], {}
    for period in PERIODS:
        targets = [r for r in rows if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in targets}):
            held = [r for r in targets if r['fit_before_date'] == cutoff]
            for arm in ARMS:
                tick = time.perf_counter()
                model = FixedMultiRatingXGB().fit(rows,date.fromisoformat(cutoff),feature_key=arm+'_features')
                p = model.predict_many([r[arm+'_features'] for r in held],
                                       [r['forecast_before_date'] for r in held])
                key = f'{arm}_{cutoff}'
                joblib.dump(model,model_dir/(key+'.joblib'),compress=3)
                params[key] = model.parameters()
                tracks[arm].extend({**{k:r[k] for k in CONTEXT},'probabilities':v,
                                    'output_scope':'1X2_only'} for r,v in zip(held,p,strict=True))
                windows.append({'period':period,'cutoff':cutoff,'arm':arm,
                    'n_training':model.parameters()['n_matches'],'n_target':len(held),
                    'seconds':time.perf_counter()-tick})
                write(OUT/'训练进度.json',{'completed_classifier_fits':len(windows),'windows':windows})
                print(json.dumps(windows[-1]|{'completed_classifier_fits':len(windows)},ensure_ascii=False),flush=True)
                del model
                gc.collect()
    for values in tracks.values():
        values.sort(key=lambda r:(r['date'],r['match_id']))
    write(OUT/'全部分类器参数.json',params)
    write(OUT/'逐场完整胜平负预测.json',tracks)
    results = {'method':method,'classifier_fits':len(windows),
               'shot_adjustment_fits':coverage['rating_metadata']['new_shot_adjustment_fits'],
               'parameter_grids':0,'by_period':{},'by_league':{},'adopted':False}
    for period in PERIODS:
        local = {arm:[r for r in values if r['period']==period] for arm,values in tracks.items()}
        results['by_period'][period] = comparison(local,intervals=True)
        results['by_league'][period] = {c:comparison({arm:[r for r in values if r['competition']==c]
                                                        for arm,values in local.items()}) for c in COMPS}
    write(OUT/'完整结果.json',results)
    report(results,coverage)
    print(json.dumps({'directory':str(OUT),'classifier_fits':len(windows),
        'scores':{p:{a:{k:s[k] for k in ('n','accuracy','log_loss','brier_sum')}
                     for a,s in d['scores'].items()} for p,d in results['by_period'].items()}},ensure_ascii=False),flush=True)


def report(result,coverage):
    lines = ['# 多路动态评级与丰富信息联合：首次完整适配实跑','',
        '本轮实际安装XGBoost并训练两个固定完整候选，不是原论文成绩复现。'
        '统计、赛程与英超可用性整体进入联合方案；没有要求每项输入先独立胜出。','',
        f"评级核心{coverage['feature_counts']['rating_core']}列，丰富联合{coverage['feature_counts']['rich_joint']}列；"
        '其余联赛没有FPL覆盖，保留未知。直接分类只输出主胜／平局／客胜；比分、总球与让球原线未改变。','',
        '| 赛季 | 方案 | 场数 | 单选命中 | LogLoss↓ | Brier↓ | 平局召回 |',
        '|---|---|---:|---:|---:|---:|---:|']
    names={'saved_performance':'原丰富表现','saved_allcomp':'原表现＋全赛事','rating_core':'评级核心＋XGBoost','rich_joint':'完整丰富联合＋XGBoost'}
    for period,d in result['by_period'].items():
        for a,s in d['scores'].items():
            lines.append(f"| {period} | {names[a]} | {s['n']} | {s['correct']}/{s['n']}（{s['accuracy']:.2%}） | {s['log_loss']:.6f} | {s['brier_sum']:.6f} | {s['class_recall']['draw']:.2%} |")
    lines += ['','## 实际接入与边界','',
        f"- 已完成{result['classifier_fits']}次固定分类器训练，另有{result['shot_adjustment_fits']}次较早年度射门转换模型训练；零参数网格。",
        '- 结果／进球Elo、pi与射门／射正强度采用作者结构及常数；射门转换由随机二折改为每年7月1日前训练，全年冻结。',
        '- 比较沿用原来的赛前3个日历日输入和28天参数更新日期；训练标签严格早于参数日期。并非精确72小时发布重建。',
        '- 原分类器未锁版本，本轮明确采用XGBoost 3.4.1、100轮／深度6等固定设置及原生缺失处理；未执行有写盘副作用的上游实验脚本。',
        '- 没有可靠时点的目标赔率、目标首发身价未进入；这不是包含原论文全部增强数据的复制。',
        '- 全部目标保留，现有预测直接复用；不拿变更训练范围或缺失删场制造提升。',
        '- 历史赛季已经多轮查看，属于前推研究，不是新的盲测。下方数值应结合联赛分布和概率质量解读。','',
        '## 本轮收束','',
        '两套完整候选、全部原日期、两季同场多指标已经完成，本轮不继续搜索深度、窗口或挑字段救分。'
        '本次结果只限定这些完整适配版本，不判定多信息、XGBoost或评级方向无价值；若有取舍，保留其任务定位。'
        '当前比赛预测、LLM与联合预测没有被替换，也没有查询当前新赛果。','',
        '逐联赛结果、校准分档、类别混淆与成对四周条件区间见同目录`完整结果.json`；全部模型在`models/`。']
    (OUT/'首次接入结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--reuse-prepared',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.reuse_prepared:
        rows = read(OUT/'联合赛前输入.json.gz')['rows']
        coverage = read(OUT/'联合接入实测.json')
    else:
        rows,coverage = prepare()
    if not args.prepare_only:
        run(rows,coverage)


if __name__ == '__main__':
    main()

"""Describe current 121-column saved forecasts; no fitting or prediction edits."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from football_model.evaluation import probability_scores, calibration_bins
from football_model.forecast_diagnostics import moving_block_means
from diagnose_1x2_discrimination_20261025 import pair_rank_auc


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'
PRIOR = ROOT/'outputs/丰富模型单温度校准_20261026T045106381635Z'
CLASS_NAMES = ('home', 'draw', 'away')
CLASS_LABELS = ('主胜', '平局', '客胜')
PERIODS = ('2024-25', '2025-26')
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
BANDS = (0., .4, .5, .6, .7, 1.0000001)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def indices(records, group):
    if group == 'all':
        return list(range(len(records)))
    if group in ('balanced', 'clear_favourite'):
        return [i for i,r in enumerate(records) if r['strength_group'] == group]
    if group in ('home_favourite', 'away_favourite', 'equal_goal_rates'):
        lookup = {'home_favourite': 0, 'away_favourite': 2, 'equal_goal_rates': None}
        return [i for i,r in enumerate(records) if r['goal_favourite_class'] == lookup[group]]
    raise ValueError('Unknown diagnostic partition')


def summary(records, arm, *, intervals=False):
    if not records:
        return {'n': 0}
    p = np.asarray([r[arm+'_probabilities'] for r in records])
    y = np.asarray([r['actual_outcome'] for r in records], dtype=int)
    truth = np.eye(3)[y]
    top = p.argmax(axis=1)
    residual = p-truth
    confusion = np.zeros((3,3), dtype=int)
    np.add.at(confusion, (y, top), 1)
    correct = top == y
    wrong = ~correct
    wrong_n = int(wrong.sum())
    class_results = {}
    for c,name in enumerate(CLASS_NAMES):
        selected = top == c
        actual = y == c
        class_results[name] = {
            'actual_n': int(actual.sum()), 'actual_frequency': float(actual.mean()),
            'mean_probability': float(p[:,c].mean()), 'mean_bias_pp': float(residual[:,c].mean()*100),
            'auc': pair_rank_auc(actual, p[:,c]), 'top1_selected_n': int(selected.sum()),
            'top1_correct_n': int((selected & actual).sum()),
            'precision_when_top1': float(actual[selected].mean()) if selected.any() else None,
            'recall': float(selected[actual].mean()) if actual.any() else None,
        }
    error_counts = {
        'actual_draw_but_selected_home_or_away': int((wrong & (y == 1)).sum()),
        'actual_decisive_but_selected_opposite_winner': int((wrong & (y != 1) & (top != 1)).sum()),
        'actual_decisive_but_selected_draw': int((wrong & (y != 1) & (top == 1)).sum()),
    }
    if sum(error_counts.values()) != wrong_n:
        raise ValueError('Error descriptions must partition all errors')
    bands = []
    max_p = p.max(axis=1)
    for lo,hi in zip(BANDS[:-1], BANDS[1:]):
        mask = (max_p >= lo) & (max_p < hi)
        n = int(mask.sum())
        bands.append({'lower_inclusive': lo, 'upper_exclusive': min(hi,1.), 'n': n,
            'mean_top1_probability': float(max_p[mask].mean()) if n else None,
            'observed_top1_accuracy': float(correct[mask].mean()) if n else None,
            'bias_pp': float((max_p[mask]-correct[mask]).mean()*100) if n else None})
    fav_rows = [r for r in records if r['goal_favourite_class'] is not None]
    favourite_events = {}
    if fav_rows:
        fp = np.asarray([r[arm+'_probabilities'] for r in fav_rows])
        fy = np.asarray([r['actual_outcome'] for r in fav_rows])
        f = np.asarray([r['goal_favourite_class'] for r in fav_rows])
        for name,classes in (('favourite_win', f), ('draw', np.ones(len(f),dtype=int)),
                             ('weaker_side_win', 2-f)):
            prediction = fp[np.arange(len(f)),classes]
            outcome = fy == classes
            favourite_events[name] = {'n': len(f), 'mean_probability': float(prediction.mean()),
                'actual_frequency': float(outcome.mean()),
                'bias_pp': float((prediction-outcome).mean()*100)}
    result = {'n': len(records), 'scores': probability_scores(p,y), 'by_class': class_results,
        'confusion_actual_rows_predicted_columns': confusion.tolist(),
        'correct_n': int(correct.sum()), 'wrong_n': wrong_n, 'error_type_counts': error_counts,
        'error_type_fraction_of_errors': {k: v/wrong_n if wrong_n else None for k,v in error_counts.items()},
        'top_probability_bands': bands, 'class_calibration_bins_5': calibration_bins(p,y,bins=5),
        'base_goal_favourite_events': favourite_events,
        'diagnostic_non_draw_actual_subset': {
            'n': int((y != 1).sum()),
            'accuracy': float(correct[y != 1].mean()) if (y != 1).any() else None,
            'note': 'Realised outcome selected this subset; cannot filter pre-match games using it.'},
        'class_bias_interval_order': list(CLASS_NAMES),
        'class_bias_pp_4week_conditional_intervals': moving_block_means(
            [r['date'] for r in records], residual*100, block_weeks=4) if intervals else None}
    return result


def section(records, *, intervals=False):
    if not records:
        return {'n': 0}
    out = {arm: summary(records, arm, intervals=intervals) for arm in ('rich', 'goal')}
    out['n'] = len(records)
    rich, goal = out['rich']['scores'], out['goal']['scores']
    out['rich_minus_goal'] = {k: rich[k]-goal[k] for k in ('log_loss','brier_sum','accuracy')}
    return out


def prepare(source, lead):
    tracks = read(source/f'lead_{lead}/逐场完整预测.json')
    pairs = {arm: {r['match_id']: r for r in tracks[name]}
             for arm,name in (('rich','fresh_performance'),('goal','goal'))}
    if (len(pairs['rich']) != len(tracks['fresh_performance']) or
            len(pairs['goal']) != len(tracks['goal']) or pairs['rich'].keys() != pairs['goal'].keys()):
        raise ValueError('Need unique matched full-cohort source forecasts')
    result = []
    for mid in sorted(pairs['rich'], key=lambda k: (pairs['rich'][k]['date'], k)):
        rich, goal = pairs['rich'][mid], pairs['goal'][mid]
        keys = ('match_id','date','period','competition','home','away','fit_before_date',
                'forecast_before_date','horizon_days','actual_score','actual_outcome')
        if any(rich[k] != goal[k] for k in keys):
            raise ValueError('Reference forecast identities differ')
        if not rich['fit_before_date'] <= rich['forecast_before_date'] < rich['date']:
            raise ValueError('Invalid earlier-information chronology')
        actual = rich['actual_score']
        if rich['actual_outcome'] != (0 if actual[0]>actual[1] else 2 if actual[0]<actual[1] else 1):
            raise ValueError('Invalid outcome axis')
        rate = np.asarray(goal['rates'], dtype=float)
        if rate.shape != (2,) or not np.isfinite(rate).all() or np.any(rate <= 0):
            raise ValueError('Positive earlier goal means required')
        strength = float(rate.max()/rate.min())
        row = {k:rich[k] for k in keys}
        row.update(goal_rate_ratio=strength, strength_group='balanced' if strength<1.5 else 'clear_favourite',
                   goal_favourite_class=0 if rate[0]>rate[1] else 2 if rate[0]<rate[1] else None,
                   goal_rates=goal['rates'], rich_rates=rich['rates'])
        for arm,old in (('rich', rich), ('goal', goal)):
            values = np.asarray(old['probabilities'], dtype=float)
            if (values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0)
                    or abs(values.sum()-1) > 1e-10):
                raise ValueError('Original class probabilities must already be valid')
            row[arm+'_probabilities'] = old['probabilities']
        result.append(row)
    return result


def make_report(out, results):
    lines = ['# 当前121项模型：胜平负偏差与错误定位', '',
        '仅复用此前保存的历史赛前预测；新增训练0次、校准0次、阈值搜索0次，未查询当前赛果或替换预测。'
        '主分析为提前3日；提前1日是重叠比赛上的口径核对，不算独立复验。', '',
        '## 两季主／平／客：概率和辨别', '',
        '| 赛季 | 结果 | 平均概率 | 实际频率 | 偏差 | AUC | 首选场数 |', '|---|---|---:|---:|---:|---:|---:|']
    for period,data in results['horizons']['3']['by_period'].items():
        for c,label in zip(CLASS_NAMES,CLASS_LABELS):
            v = data['overall']['rich']['by_class'][c]
            lines.append(f"| {period} | {label} | {v['mean_probability']:.2%} | {v['actual_frequency']:.2%} | "
                         f"{v['mean_bias_pp']:+.2f}个百分点 | {v['auc']:.4f} | {v['top1_selected_n']} |")
    lines += ['', 'AUC衡量该结果发生的比赛能否通常排得更高，0.5对应无排序辨别；它不是命中率。'
              '平均概率偏差为预测减实际，不能直接把同样百分点加到未来比赛上。', '',
        '## 单选错误的构成', '',
        '| 赛季 | 正确／全部 | 错误数 | 实际平局却选胜负 | 实际胜负却选反 | 实际胜负却选平 |',
        '|---|---:|---:|---:|---:|---:|']
    for period,data in results['horizons']['3']['by_period'].items():
        v = data['overall']['rich']; e = v['error_type_counts']
        lines.append(f"| {period} | {v['correct_n']}/{v['n']} | {v['wrong_n']} | "
                     f"{e['actual_draw_but_selected_home_or_away']} | "
                     f"{e['actual_decisive_but_selected_opposite_winner']} | "
                     f"{e['actual_decisive_but_selected_draw']} |")
    lines += ['', '这是已发生错误的分类，不是失败原因的因果分解。事后只取非平局比赛可得到更高命中率，'
              '但比赛前不知道哪些会平，不能把该子集当成可执行筛选法。', '',
        '## 沿用既有强弱分组', '',
        '用原goal基专家的较高／较低预期进球之比：小于1.5为均势，否则为明显强弱。'
        '该定义来自旧研究，未为本次结果挑阈值。热门是基专家预期进球较高的一侧。', '',
        '| 赛季 | 分组 | 场数 | 原goal LL | 丰富模型LL | 单选命中率 | 热门胜偏差 | 平局偏差 | 较弱方胜偏差 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for period,data in results['horizons']['3']['by_period'].items():
        for name,label in (('balanced','均势'),('clear_favourite','明显强弱')):
            s = data['groups'][name]; v = s['rich']; e=v['base_goal_favourite_events']
            lines.append(f"| {period} | {label} | {v['n']} | {s['goal']['scores']['log_loss']:.6f} | "
                         f"{v['scores']['log_loss']:.6f} | {v['scores']['accuracy']:.2%} | "
                         f"{e['favourite_win']['bias_pp']:+.2f}pp | {e['draw']['bias_pp']:+.2f}pp | "
                         f"{e['weaker_side_win']['bias_pp']:+.2f}pp |")
    lines += ['', '## 相同类别偏差的时间波动', '',
        '| 赛季 | 结果 | 偏差的4周条件区间（百分点） |', '|---|---|---|']
    for period,data in results['horizons']['3']['by_period'].items():
        v=data['overall']['rich']
        for label,ci in zip(CLASS_LABELS,v['class_bias_pp_4week_conditional_intervals']['interval_95']):
            lines.append(f'| {period} | {label} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |')
    lines += ['', '该区间是在既有预测条件下的配对日历分块描述，不包含重新训练、所有参数估计和历史资料修订的不确定性。'
              '不同分组与两提前量相互重叠，不是多份独立证据。', '',
        '## 提前1日核对：全部列出', '',
        '| 赛季 | 场数 | 主胜偏差 | 平局偏差 | 客胜偏差 | 平局AUC | 平局首选场数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for period,data in results['horizons']['1']['by_period'].items():
        s=data['overall']['rich']; a=s['by_class']
        lines.append(f"| {period} | {s['n']} | {a['home']['mean_bias_pp']:+.2f}pp | "
                     f"{a['draw']['mean_bias_pp']:+.2f}pp | {a['away']['mean_bias_pp']:+.2f}pp | "
                     f"{a['draw']['auc']:.4f} | {a['draw']['top1_selected_n']} |")
    lines += ['', '## 五联赛描述：保留两季', '',
        '| 赛季／联赛 | 场数 | 主胜偏差 | 平局偏差 | 客胜偏差 | 平局AUC |', '|---|---:|---:|---:|---:|---:|']
    for period,data in results['horizons']['3']['by_period'].items():
        for comp,s in data['by_league'].items():
            a=s['rich']['by_class']
            lines.append(f"| {period}／{comp} | {s['n']} | {a['home']['mean_bias_pp']:+.2f}pp | "
                         f"{a['draw']['mean_bias_pp']:+.2f}pp | {a['away']['mean_bias_pp']:+.2f}pp | {a['draw']['auc']:.4f} |")
    lines += ['', '## 使用边界与停止点', '',
        '本轮只定位当前121项模型的错误，不重跑旧161项分类研究，不再尝试温度、类别偏置或其他校准。'
        '主客偏差、平局不足和强弱压缩可能同时存在；仅凭这些分组不能把原因归于近期战绩、独立比分假设、'
        '参数收缩或伤停遗漏中的某一项。', '',
        '完整三类混淆矩阵、固定概率分箱、goal参照以及所有分组见完整诊断结果.json。'
        '该诊断不自动产生新比赛概率、提高命中率或变更投注方向。']
    (out/'偏差与错误定位.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    now=datetime.now(timezone.utc)
    out=ROOT/'outputs'/now.strftime('当前121模型胜平负定位_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True)
    method={'created_at':now.isoformat(),'source':str(SOURCE),'candidate':'fresh_performance121',
        'primary_horizon_days':3,'sensitivity_horizon_days':1,'periods':list(PERIODS),
        'group_definition':'Reuse goal rate ratio<1.5 balanced, otherwise clear favourite; no new chosen threshold',
        'metrics':['1X2 LogLoss/Brier/top1','three-class mean prediction minus actual frequency',
                   'actual-row predicted-column confusion','one-versus-rest AUC',
                   'existing fixed .2 class bins','top forecast probability bands0/.4/.5/.6/.7/1'],
        'uncertainty':'Conditional 4-calendar-week moving blocks 2000 draws, same existing routine',
        'scope':'Already viewed historical seasons; no fresh blind validation or causal error attribution',
        'reuse':'Existing pair_rank_auc/evaluation/moving_block_means; verify prior late class bias result',
        'stopping_point':'Finish exactly both saved horizons and both seasons; no fitting, calibration, threshold '
                         'selection, new prospective outcome query or prediction replacement',
        'new_fits':0,'calibration_fits':0,'searches':0,'current_results_checked':False,'forecast_changes':False}
    write(out/'方法与停止点.json',method)
    result={'method':method,'horizons':{}}
    for lead in (3,1):
        records=prepare(SOURCE,lead)
        section_by_period={}
        for period in PERIODS:
            local=[r for r in records if r['period']==period]
            groups={g:section([local[i] for i in indices(local,g)],intervals=True)
                    for g in ('balanced','clear_favourite','home_favourite','away_favourite','equal_goal_rates')}
            overall=section(local,intervals=True)
            previous=read(PRIOR/f'提前{lead}日_完整结果.json')
            if period=='2025-26':
                current=[overall['rich']['by_class'][c]['mean_bias_pp'] for c in CLASS_NAMES]
                if not np.allclose(current,previous['overall']['class_bias_pp']['original'],rtol=0,atol=1e-10):
                    raise ValueError('Reused late-period class bias differs from existing result')
            section_by_period[period]={'overall':overall,'groups':groups,
                'by_league':{c:section([r for r in local if r['competition']==c]) for c in COMPS}}
        result['horizons'][str(lead)]={'n':len(records),'by_period':section_by_period}
        write(out/f'提前{lead}日_已有预测与固定分组.json',{'records':records})
    write(out/'完整诊断结果.json',result)
    make_report(out,result)
    print(json.dumps({'output':str(out),'new_fits':0,'searches':0},ensure_ascii=False),flush=True)
    for lead,body in result['horizons'].items():
        for period,s in body['by_period'].items():
            v=s['overall']['rich']
            print(json.dumps({'lead':lead,'period':period,'n':v['n'],'scores':v['scores'],
                'classes':v['by_class'],'errors':v['error_type_counts'],
                'groups':{k:{'n':g['n'],'rich_scores':g['rich']['scores'],
                    'goal_events':g['rich']['base_goal_favourite_events']} for k,g in s['groups'].items() if g['n']}
                },ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

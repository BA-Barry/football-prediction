"""Describe stored v0.49 probabilities without fitting, replay or selection.

The log-loss identity separates draw/non-draw identification from direction
conditional on a non-draw. It does not define a new prediction or branch rule.
All output and computation artifacts remain on G:.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT / 'outputs/持续状态净胜球研究_20261028'
SOURCE = OUT / '逐场概率与状态.json.gz'
NAMES = ('主胜', '平局', '客胜')
TRACKS = {'original': 'original_hda', 'persistent': 'gas_probabilities'}
EDGES = np.asarray([0., .2, .4, .6, .8, 1.])


def mean(values):
    return float(np.mean(values)) if len(values) else None


def quantiles(values):
    return {str(q): float(v) for q, v in zip((0., .1, .25, .5, .75, .9, 1.),
        np.quantile(values, [0., .1, .25, .5, .75, .9, 1.]))} if len(values) else {}


def arrays(rows, field):
    p = np.asarray([r[field] for r in rows], dtype=float)
    if p.shape != (len(rows), 3) or not np.all(np.isfinite(p)):
        raise ValueError(f'Need complete stored three-probability rows: {field}')
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError(f'Zero/unit probabilities need explicit infinite-loss treatment, not clipping: {field}')
    if not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-12):
        raise ValueError(f'Probability sums differ from one: {field}')
    y = np.asarray([0 if r['hg'] > r['ag'] else 2 if r['hg'] < r['ag'] else 1 for r in rows])
    return p, y


def loss_parts(p, y):
    draw = y == 1
    not_draw = ~draw
    non_draw_mass = p[:,0] + p[:,2]
    full = -np.log(p[np.arange(len(y)), y])
    # Use log(1-pD) for the binary problem; compare its exact algebraic identity
    # with full loss. Tiny floating-point sum discrepancies remain measured.
    binary = np.where(draw, -np.log(p[:,1]), -np.log1p(-p[:,1]))
    conditional = np.zeros(len(y), dtype=float)
    conditional[not_draw] = -np.log(p[np.arange(len(y))[not_draw], y[not_draw]] / non_draw_mass[not_draw])
    residual = full - binary - conditional
    if np.max(np.abs(residual), initial=0.) > 1e-12:
        raise ValueError('Draw/non-draw + conditional direction identity failed')
    return full, binary, conditional, residual


def bins(probability, observed):
    assigned = np.minimum(np.searchsorted(EDGES, probability, side='right')-1, len(EDGES)-2)
    out = []
    for i in range(len(EDGES)-1):
        mask = assigned == i
        count = int(mask.sum())
        out.append({'lower': float(EDGES[i]), 'upper': float(EDGES[i+1]),
                    'right_closed': i == len(EDGES)-2, 'n': count,
                    'mean_probability': mean(probability[mask]), 'observed_rate': mean(observed[mask]),
                    'mean_probability_minus_observed_rate': mean(probability[mask]-observed[mask])})
    if sum(r['n'] for r in out) != len(probability):
        raise ValueError('Probability bins lost rows')
    return out


def metrics(p, y):
    full, binary, conditional, residual = loss_parts(p, y)
    draw = y == 1
    non_draw = ~draw
    q_home = p[:,0] / (p[:,0] + p[:,2])
    return {'n': len(y), 'log_loss': mean(full), 'draw_binary_log_loss': mean(binary),
            'non_draw_weight': float(non_draw.mean()),
            'conditional_direction_log_loss_on_actual_non_draws': mean(conditional[non_draw]),
            'conditional_direction_weighted_log_loss': mean(conditional),
            'identity_max_abs_residual': float(np.max(np.abs(residual), initial=0.)),
            'draw_auc': float(roc_auc_score(draw, p[:,1])) if len(set(draw)) == 2 else None,
            'conditional_home_away_auc_on_actual_non_draws': float(roc_auc_score(y[non_draw] == 0, q_home[non_draw])) if len(set(y[non_draw])) == 2 else None,
            'conditional_home_away_correct_on_actual_non_draws': int(((q_home >= .5) & (y == 0) | (q_home < .5) & (y == 2)).sum()),
            'conditional_scope': 'Actual non-draws are selected only for retrospective direction diagnosis. This cannot select non-draw matches before kickoff.',
            'by_outcome': {name: {'predicted_mean': float(p[:,i].mean()), 'actual_rate': float((y == i).mean()),
                                 'mean_bias': float(p[:,i].mean()-(y == i).mean()),
                                 'class_brier': float(np.mean((p[:,i]-(y == i))**2)),
                                 'fixed_bins': bins(p[:,i], y == i)} for i, name in enumerate(NAMES)},
            'conditional_home_probability_bins_actual_non_draws': bins(q_home[non_draw], y[non_draw] == 0),
            'minimum_saved_probability': float(p.min()), 'probability_clip_rows': 0}


def difference(p0, p1, y):
    f0, b0, c0, _ = loss_parts(p0, y)
    f1, b1, c1, _ = loss_parts(p1, y)
    delta_full, delta_binary, delta_conditional = map(mean, (f1-f0, b1-b0, c1-c0))
    if abs(delta_full-delta_binary-delta_conditional) > 1e-12:
        raise ValueError('Aggregate difference identity failed')
    return {'new_minus_old_full_log_loss': delta_full,
            'new_minus_old_draw_binary_log_loss': delta_binary,
            'new_minus_old_weighted_conditional_direction_log_loss': delta_conditional,
            'meaning': 'Positive means persistent line assigns less probability to observed outcome on average; this is an exact accounting identity, not causal attribution.'}


def transitions(p0, p1, y):
    old, new = p0.argmax(axis=1), p1.argmax(axis=1)
    f0, b0, c0, _ = loss_parts(p0, y)
    f1, b1, c1, _ = loss_parts(p1, y)
    out = []
    for a in range(3):
        for b in range(3):
            mask = (old == a) & (new == b)
            n = int(mask.sum())
            out.append({'old_choice': NAMES[a], 'new_choice': NAMES[b], 'n': n, 'changed': a != b,
                'actual_outcomes': {name: int((mask & (y == i)).sum()) for i, name in enumerate(NAMES)},
                'old_correct': int((mask & (old == y)).sum()), 'new_correct': int((mask & (new == y)).sum()),
                'corrected': int((mask & (old != y) & (new == y)).sum()),
                'harmed': int((mask & (old == y) & (new != y)).sum()),
                'both_wrong': int((mask & (old != y) & (new != y)).sum()),
                'old_log_loss_sum': float(f0[mask].sum()), 'new_log_loss_sum': float(f1[mask].sum()),
                'log_loss_delta_sum': float((f1-f0)[mask].sum()),
                'contribution_to_full_cohort_mean_LL_difference': float((f1-f0)[mask].sum()/len(y)),
                'draw_binary_LL_delta_sum': float((b1-b0)[mask].sum()),
                'weighted_direction_LL_delta_sum': float((c1-c0)[mask].sum())})
    if sum(r['n'] for r in out) != len(y):
        raise ValueError('Transition rows do not reconcile')
    return out


def net_goal_diagnostics(rows, p0, p1, y):
    rates = np.exp(np.asarray([r['log_rates'] for r in rows], dtype=float))
    if rates.shape != (len(rows), 2) or not np.all(np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError('Need valid stored latent rates for all rows')
    actual = np.asarray([r['hg']-r['ag'] for r in rows])
    predicted_mean = rates[:,0]-rates[:,1]
    predicted_variance = rates.sum(axis=1)
    residual = actual-predicted_mean
    standardized = residual/np.sqrt(predicted_variance)
    old_choice, new_choice = p0.argmax(axis=1), p1.argmax(axis=1)
    direction_flips = []
    for old, new in ((2,0), (0,2), (0,1), (2,1)):
        mask = (old_choice == old) & (new_choice == new)
        direction_flips.append({'old_choice': NAMES[old], 'new_choice': NAMES[new], 'n': int(mask.sum()),
            'mean_predicted_net_goals': mean(predicted_mean[mask]), 'mean_actual_net_goals': mean(actual[mask]),
            'mean_residual_actual_minus_predicted': mean(residual[mask]),
            'mean_predicted_variance': mean(predicted_variance[mask]),
            'standardized_residual_mean': mean(standardized[mask]),
            'actual_outcomes': {name: int((mask & (y == i)).sum()) for i, name in enumerate(NAMES)}})
    intervals = [(-math.inf, -1.5), (-1.5, -.5), (-.5, .5), (.5, 1.5), (1.5, math.inf)]
    net_bins = []
    for lo, hi in intervals:
        mask = (predicted_mean >= lo) & (predicted_mean < hi)
        net_bins.append({'lower': None if not math.isfinite(lo) else lo,
                        'upper': None if not math.isfinite(hi) else hi, 'n': int(mask.sum()),
                        'mean_prediction': mean(predicted_mean[mask]), 'actual_mean': mean(actual[mask]),
                        'mean_residual': mean(residual[mask]),
                        'mean_squared_residual': mean(residual[mask]**2),
                        'mean_model_variance': mean(predicted_variance[mask])})
    return {'n': len(rows), 'mean_predicted_net_goals': mean(predicted_mean), 'mean_actual_net_goals': mean(actual),
            'mean_residual_actual_minus_predicted': mean(residual),
            'mean_model_variance': mean(predicted_variance), 'mean_squared_residual': mean(residual**2),
            'standardized_residual_mean': mean(standardized), 'standardized_residual_second_moment': mean(standardized**2),
            'net_goal_residual_quantiles': quantiles(residual),
            'predicted_mean_net_goal_bins': net_bins, 'direction_transition_groups': direction_flips,
            'limit': 'The rates are latent Skellam distribution parameters, not real expected goals/xG. Residual summaries are diagnostics on saved historical predictions; no parameter is estimated here.'}


def summarize(rows):
    p0, y = arrays(rows, 'original_hda')
    p1, y1 = arrays(rows, 'gas_probabilities')
    if not np.array_equal(y, y1):
        raise ValueError('Target labels changed across tracks')
    return {'n': len(rows), 'original': metrics(p0,y), 'persistent': metrics(p1,y),
            'difference': difference(p0,p1,y), 'transitions': transitions(p0,p1,y),
            'net_goal_diagnostics': net_goal_diagnostics(rows,p0,p1,y)}


def render(summary):
    overall = summary['overall']
    delta = overall['difference']
    rows = []
    for period, s in [('全部', overall), *summary['by_season'].items()]:
        for track, name in (('original','原丰富模型'), ('persistent','持续净胜球候选')):
            m = s[track]
            rows.append(f"|{period}|{name}|{m['n']}|{m['log_loss']:.6f}|{m['draw_binary_log_loss']:.6f}|{m['conditional_direction_weighted_log_loss']:.6f}|{m['conditional_direction_log_loss_on_actual_non_draws']:.6f}|{m['draw_auc']:.4f}|{m['conditional_home_away_auc_on_actual_non_draws']:.4f}|")
    transition_rows = []
    for r in overall['transitions']:
        a = r['actual_outcomes']
        transition_rows.append(f"|{r['old_choice']}→{r['new_choice']}|{r['n']}|{a['主胜']}/{a['平局']}/{a['客胜']}|{r['corrected']}|{r['harmed']}|{r['both_wrong']}|{r['contribution_to_full_cohort_mean_LL_difference']:+.6f}|")
    net_rows=[]
    for period,s in [('全部',overall),*summary['by_season'].items()]:
        n=s['net_goal_diagnostics']
        net_rows.append(f"|{period}|{n['mean_predicted_net_goals']:.4f}|{n['mean_actual_net_goals']:.4f}|{n['mean_residual_actual_minus_predicted']:+.4f}|{n['mean_model_variance']:.4f}|{n['mean_squared_residual']:.4f}|{n['standardized_residual_second_moment']:.4f}|")
    original,persistent=overall['original'],overall['persistent']
    nd=overall['net_goal_diagnostics']
    total_bad = sum(r['harmed'] for r in overall['transitions'])
    total_good = sum(r['corrected'] for r in overall['transitions'])
    shares = summary['accounting_shares']
    league_rows=[]
    for league,s in summary['by_competition'].items():
        n=s['net_goal_diagnostics']
        league_rows.append(f"|{league}|{s['n']}|{s['difference']['new_minus_old_full_log_loss']:+.6f}|{s['persistent']['by_outcome']['主胜']['mean_bias']*100:+.2f}|{n['mean_residual_actual_minus_predicted']:+.4f}|")
    text=f'''# 持续净胜球候选：概率损失来源分解

2026-10-28。仅分析已保存的3,452场三概率和潜在率；**训练、前推重放、参数调整、重拼候选、赛果查询均为0**。本页补充上一轮为何退步的证据，不宣布另一种已改进模型。

## 精确分解：问题位于哪里

每场胜平负对数损失可写为：**平／非平二分类损失＋实际非平时的主／客条件损失**。总体第二项已经乘实际非平占比；表中另列未加权条件平均，避免把只筛实际非平后的成绩当可执行预测。概率不截断、不重新归一化。

|范围|线路|场数|完整LL↓|平／非平LL↓|条件方向加权LL↓|非平样本条件LL↓|平局AUC|非平样本H/A AUC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

总体新减旧LL为{delta['new_minus_old_full_log_loss']:+.6f}，精确拆为平／非平{delta['new_minus_old_draw_binary_log_loss']:+.6f}和条件主客方向{delta['new_minus_old_weighted_conditional_direction_log_loss']:+.6f}。正数表示变差；这是一种记账恒等式，并非说明某一个参数造成了这些变化。逐场恒等误差最大{summary['checks']['maximum_identity_residual']:.3g}；与上一轮保存评分最大差{summary['checks']['maximum_saved_summary_LL_difference']:.3g}。

其中约{shares['conditional_direction_share_of_positive_total_LL_increase']:.1%}的总体LL增加来自条件方向项。两季的方向项分别为{summary['by_season']['2024/25']['difference']['new_minus_old_weighted_conditional_direction_log_loss']:+.6f}和{summary['by_season']['2025/26']['difference']['new_minus_old_weighted_conditional_direction_log_loss']:+.6f}，不能把本轮退步仅归为“平局分布不够好”。

## 哪些改向获得或损失收益

|原首选→新首选|场数|实际主／平／客|纠正|改坏|两者均错|对总体LL差贡献|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(transition_rows)}

纠正共{total_good}、改坏共{total_bad}。未改首选的场次也可能显著改变概率损失，因此不能只解释421场改向，更不能赛后选每场答对的线路作为新方案。

首选未变化的{shares['unchanged_top1_n']}场，对总体LL差贡献{shares['unchanged_top1_contribution_to_mean_LL_delta']:+.6f}，占本轮总恶化约{shares['unchanged_top1_share_of_positive_total_LL_increase']:.1%}。这部分准确率完全没变，但真实结果获得的概率总体更低；只追查改向样本会遗漏主要概率质量损失。

## 平均概率、分箱与净胜球残差

平均平局概率原{original['by_outcome']['平局']['predicted_mean']:.4%}、新{persistent['by_outcome']['平局']['predicted_mean']:.4%}、实际{original['by_outcome']['平局']['actual_rate']:.4%}；对应平局AUC原{original['draw_auc']:.4f}、新{persistent['draw_auc']:.4f}。平均比例接近与逐场辨别是不同问题。主、平、客各类固定五档[0,.2,.4,.6,.8,1]的样本数、平均概率、实际频率与偏差见JSON；未按观察结果另选边界或训练校准器。

|范围|平均预测净胜球|实际净胜球|实际－预测|平均模型方差|平均平方残差|标准化残差二阶矩|
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(net_rows)}

这里净胜球均值为θH−θA、方差为θH＋θA。**两个θ只是Skellam潜在率，不能叫真实xG。**二阶残差与方差的差异可能包含均值偏差、样本变异与模型结构误差，不能仅据此认定需要扩大方差。总体预测净胜球与实际差{nd['mean_residual_actual_minus_predicted']:+.4f}；各联赛、固定净胜球范围和方向改向组的残差均保存在JSON。

|联赛|场数|新减旧LL|新线主胜平均偏差（百分点）|实际－预测净胜球|
|---|---:|---:|---:|---:|
{chr(10).join(league_rows)}

总体新线平均主胜概率高估{persistent['by_outcome']['主胜']['mean_bias']*100:.2f}个百分点、客胜低估{-persistent['by_outcome']['客胜']['mean_bias']*100:.2f}个百分点，但各联赛偏差方向并不一致。**本页不能支持统一给主场扣一个固定比例**，也没有据此增加联赛修正。

## 本次证据的用途

- 应分别看平／非平辨别与主客方向；平均平局比例更接近，不能掩盖条件方向或概率损失恶化。
- 旧客胜改为主胜、旧主胜改为客胜须结合真实结果与连续概率变化核对，不能以新增平局数抵销或淡化方向错误。
- 下一项实施若保留持续状态信息，应首先保留现有丰富输入并清楚定义它加入哪个概率环节；本报告没有证明如何择路或融合才会有效。

本轮没有根据这些已知结果学习任何选择规则、阈值、校准或融合权重。后两季已反复参与研发，以上仅为历史诊断；实际非平子集使用赛后标签，只能定位方向能力，不能声称赛前准确率。一次接法的结果不否定整个状态模型研究方向。
'''
    (OUT/'概率损失来源分解.md').write_text(text,encoding='utf-8')


def main():
    data=json.load(gzip.open(SOURCE,'rt',encoding='utf-8'))
    rows=data['records']
    if len(rows)!=3452 or len({r['match_id'] for r in rows})!=3452:
        raise ValueError('Need the complete unique saved cohort')
    if not all(r['gas_available'] for r in rows):
        raise ValueError('This diagnosis requires same-coverage complete stored GAS probabilities')
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'source':str(SOURCE),'n':len(rows),'unique_match_ids':len({r['match_id'] for r in rows}),
             'fit_calls':0,'forecast_replay_calls':0,'parameter_adjustments':0,'current_results_queries':0,
             'overall':summarize(rows),
             'by_season':{s:summarize([r for r in rows if r['season']==s]) for s in sorted({r['season'] for r in rows})},
             'by_competition':{c:summarize([r for r in rows if r['competition']==c]) for c in sorted({r['competition'] for r in rows})},
             'limits':['No new predictions or model-selection/fusion rules are constructed.',
                       'Previously reviewed seasons remain development data.',
                       'Conditional H/A scope uses actual non-draw labels only for retrospective diagnosis.',
                       'Skellam latent rates are not true xG.']}
    stored=json.loads((OUT/'完整结果与分解.json').read_text(encoding='utf-8'))
    checks=[]
    for group,new,old in [('overall',summary['overall'],stored['full_cohort']),
        *[(s,summary['by_season'][s],stored['by_season'][s]) for s in summary['by_season']],
        *[(c,summary['by_competition'][c],stored['by_competition'][c]) for c in summary['by_competition']]]:
        for track,field in [('original','original'),('persistent','deployment')]:
            error=abs(new[track]['log_loss']-old[field]['log_loss'])
            if error>1e-12:
                raise ValueError('Saved full-score loss did not reproduce')
            checks.append({'scope':group,'track':track,'saved_LL_difference':error})
    identity=max(summary['overall'][track]['identity_max_abs_residual'] for track in TRACKS)
    summary['checks']={'maximum_identity_residual':identity,
                       'maximum_saved_summary_LL_difference':max(r['saved_LL_difference'] for r in checks),
                       'saved_summary_reconciliation':checks,
                       'transition_n_reconciles':sum(r['n'] for r in summary['overall']['transitions'])==len(rows),
                       'exact_positive_probabilities_only':True,'clipping_or_renormalization':False}
    overall=summary['overall']
    total_delta=overall['difference']['new_minus_old_full_log_loss']
    unchanged=[r for r in overall['transitions'] if not r['changed']]
    unchanged_delta=sum(r['contribution_to_full_cohort_mean_LL_difference'] for r in unchanged)
    summary['accounting_shares']={
        'conditional_direction_share_of_positive_total_LL_increase':
            overall['difference']['new_minus_old_weighted_conditional_direction_log_loss']/total_delta if total_delta>0 else None,
        'unchanged_top1_n':sum(r['n'] for r in unchanged),
        'unchanged_top1_contribution_to_mean_LL_delta':unchanged_delta,
        'unchanged_top1_share_of_positive_total_LL_increase':unchanged_delta/total_delta if total_delta>0 else None,
        'scope':'Shares describe this fixed realized positive loss increase, not causal parameter effects or learned weights.'}
    (OUT/'概率损失来源分解.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    render(summary)
    print(json.dumps({'n':summary['n'],'difference':summary['overall']['difference'],
                      'checks':summary['checks'],'net_goal_diagnostics':summary['overall']['net_goal_diagnostics']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

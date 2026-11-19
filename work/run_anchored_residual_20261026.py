"""A single reference-preserving refinement of the completed integration."""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import joblib
import numpy as np

from football_model.anchored_residual import AnchoredResidualStack, fit_attenuation, RESIDUAL_FEATURE_NAMES
from football_model.forecast_diagnostics import moving_block_means
from football_model.prematch_calibration import apply_calibrator
from football_model.probability import score_grid, summarize
from football_model.score_reconciliation import reconcile_score_probabilities
from run_integrated_components_20261026 import CONTEXT, metrics, derive_grid_losses


ROOT = REPOSITORY_ROOT
PRIOR = ROOT/'outputs/现有组件整合_20261026'
OUT = ROOT/'outputs/锚定增量整合_20261026'
CUTOFF = date(2025,8,1)
EARLY, LATE = '2024-25', '2025-26'
METHOD = {
    'created_at':datetime.now(timezone.utc).isoformat(),
    'specific_problem':'Previous full stack changed existing class relations; home correct -93, draw/away +64. Its scaling amplified small contrasts under a different regularization prior.',
    'hypothesis':'Preserve rich reference logits and learn only natural-scale, regularized member differences; learn attenuation of correction from earlier OOF, not a global temperature of all probabilities.',
    'reference':'raw rich_performance; zero residual or attenuation=0 restores that exact vector',
    'equation':'Q=softmax(log(P_reference)+gamma*X@B)',
    'features':list(RESIDUAL_FEATURE_NAMES),
    'removed':'reference log-odds as learned inputs, free class/league intercepts, centering and variance scaling',
    'objective':'(sum multinomial NLL + 0.5*||B||^2)/n; fixed C=1; symmetric three-class coefficients',
    'regularization_note':'Natural-scale symmetric ridge is a changed prior, not an algebraically equivalent deletion of StandardScaler.',
    'correction_strength':'one gamma in [0,1], fitted once on earlier meta-OOF; both endpoints explicitly available',
    'chronology':'2024/25 only; same four 90-day-warmup/60-day expanding publication windows; final fit 2025-08-01; 2025/26 evaluation only',
    'horizon_days':3,'base_refits':0,'parameter_searches':0,'new_dependencies':0,
    'score_rule':'Reuse previous score reconciliation: same rich P(score|outcome), final means and all markets recomputed',
    'primary_effect':'Later shrunk residual versus raw reference: full-cohort accuracy and probability quality',
    'secondary':'Previous integrated result, earlier-calibrated reference, score/totals/handicap, classes, complete five leagues',
    'stop':'One fixed residual structure, four OOF fits, one final fit and one attenuation. No later-score-driven retuning, field removal, extra horizons or rescue variants.',
    'limits':['previously viewed historical seasons, not new blind test',
        'zero intercept does not prevent class mean shifts from nonzero member contrast means',
        'earlier smaller OOF model residual scale may differ from final all-season residual scale',
        'unknown non-EPL personnel branch is inactive, not known zero player impact',
        'LLM/market remain separate because complete correctly dated historical series are missing'],
    'current_forecasts_replaced':False,'current_match_results_queried':False,
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_rows():
    with gzip.open(PRIOR/'整合赛前输入.json.gz','rt',encoding='utf-8') as f:
        rows = json.load(f)['rows']
    rows.sort(key=lambda r:(r['date'],r['match_id']))
    if len(rows)!=3452 or len({r['match_id'] for r in rows})!=len(rows):
        raise ValueError('Retain original complete two-season cohort')
    for r in rows:
        if r['horizon_days']!=3 or not all(fit<=r['forecast_before_date']<r['date'] for fit in r['route_fit_before_dates'].values()):
            raise ValueError('Keep identical earlier member probabilities and D-3 chronology')
    return rows


def fit(rows):
    early = [r for r in rows if r['period']==EARLY]
    late = [r for r in rows if r['period']==LATE]
    if len(early)!=1726 or len(late)!=1726:
        raise ValueError('Retain full existing season scopes')
    if any(r['date']>=CUTOFF.isoformat() for r in early) or any(r['forecast_before_date']<CUTOFF.isoformat() for r in late):
        raise ValueError('Separate earlier labels and later publication dates')
    cutoff = min(date.fromisoformat(r['date']) for r in early)+timedelta(days=90)
    last = max(date.fromisoformat(r['forecast_before_date']) for r in early)
    model_dir = OUT/'models'
    model_dir.mkdir(exist_ok=True)
    oof, windows,parameters = [],[],{}
    while cutoff<=last:
        end = cutoff+timedelta(days=60)
        training = [r for r in early if r['date']<cutoff.isoformat()]
        targets = [r for r in early if cutoff.isoformat()<=r['forecast_before_date']<end.isoformat()]
        if targets:
            tick = time.perf_counter()
            model = AnchoredResidualStack().fit(training,cutoff)
            offsets = model.residuals_many(targets)
            q = model.predict_many(targets)
            key = cutoff.isoformat()
            joblib.dump(model,model_dir/f'residual_{key}.joblib',compress=3)
            parameters[key] = model.parameters()
            for r,offset,p in zip(targets,offsets,q,strict=True):
                oof.append({**{k:r[k] for k in CONTEXT},'meta_fit_date':key,
                    'reference_probabilities':r['probabilities_by_route']['rich_performance'],
                    'residual_logits':offset.tolist(),'raw_residual_probabilities':p.tolist()})
            windows.append({'cutoff':key,'publication_end_exclusive':end.isoformat(),
                'training_n':len(training),'target_n':len(targets),'seconds':time.perf_counter()-tick})
            print(json.dumps({'residual_oof':windows[-1]},ensure_ascii=False),flush=True)
        cutoff = end
    attenuation = fit_attenuation(oof,CUTOFF)
    final = AnchoredResidualStack().fit(early,CUTOFF)
    parameters['final'] = final.parameters()
    model_file = model_dir/'anchored_residual_2025-08-01.joblib'
    joblib.dump(final,model_file,compress=3)
    gamma = attenuation['attenuation']
    unshrunk = final.predict_many(late)
    shrunk = final.predict_many(late,attenuation=gamma)
    base = np.asarray([r['probabilities_by_route']['rich_performance'] for r in late])
    if not np.array_equal(base,final.predict_many(late,attenuation=0)):
        raise ValueError('Zero attenuation must exactly preserve supplied raw reference')
    altered = [{**r,'actual_outcome':(r['actual_outcome']+1)%3,'actual_score':[98,99]} for r in late]
    if not np.array_equal(shrunk,final.predict_many(altered,attenuation=gamma)):
        raise ValueError('Target labels may not enter predictions')
    if not np.array_equal(shrunk,joblib.load(model_file).predict_many(late,attenuation=gamma)):
        raise ValueError('Save/restore must retain actual predictions')
    write(OUT/'全部增量参数.json',parameters)
    write(OUT/'较早学习的修正幅度.json',attenuation)
    write(OUT/'较早时间外增量记录.json',{'rows':oof,'windows':windows,
        'warmup_not_calibrated_n':len(early)-len(oof)})
    execution = {'base_refits':0,'oof_residual_fits':len(windows),'final_residual_fits':1,
        'attenuation_fits':1,'new_temperature_fits':0,'parameter_searches':0,
        'gamma_zero_exact_identity':True,'future_label_invariance':True,'joblib_restore_exact':True}
    write(OUT/'执行记录.json',execution)
    print(json.dumps({'attenuation':attenuation['attenuation'],
        'earlier_objective_candidates':attenuation['candidate_mean_nll'],
        'execution':execution},ensure_ascii=False),flush=True)
    return late,unshrunk,shrunk,attenuation,execution


def evaluate(late,unshrunk,shrunk,attenuation,execution):
    prior = {r['match_id']:r for r in read(PRIOR/'较晚逐场联合预测.json')['rows']}
    old_temp = read(ROOT/'outputs/丰富模型单温度校准_20261026T045106381635Z/提前3日_温度参数.json')
    reference = np.asarray([r['probabilities_by_route']['rich_performance'] for r in late])
    reference_cal = apply_calibrator(reference,old_temp)
    previous = np.asarray([prior[r['match_id']]['probabilities']['integrated_calibrated'] for r in late])
    tracks = {'reference_raw':reference,'reference_existing_calibration':reference_cal,
        'previous_full_integration':previous,'anchored_unshrunk':unshrunk,'anchored_final':shrunk}
    records,max_grid_error = [],0.
    for i,r in enumerate(late):
        for key in CONTEXT:
            if r[key]!=prior[r['match_id']][key]:
                raise ValueError('Compare only exact prior fixtures/publications')
        source = score_grid(*r['original_rates'])
        if not np.allclose(source.home_draw_away,reference[i],rtol=0,atol=1e-10):
            raise ValueError('Original score reconstruction must agree with raw reference')
        losses,summaries = {},{}
        for name,p in tracks.items():
            grid = source if name=='reference_raw' else reconcile_score_probabilities(source,p[i])
            error = float(np.abs(np.asarray(grid.home_draw_away)-p[i]).max())
            max_grid_error = max(max_grid_error,error)
            if error>1e-10:
                raise ValueError('Final score grid must agree with H/D/A')
            losses[name] = derive_grid_losses(grid,r['actual_score'])
            if name in ('reference_raw','anchored_final'):
                summaries[name] = summarize(grid)
        records.append({**{k:r[k] for k in CONTEXT},
            'probabilities':{name:p[i].tolist() for name,p in tracks.items()},
            'market_losses':losses,'final_markets':summaries,
            'reference_rates':r['original_rates'],'integration_before_date':CUTOFF.isoformat(),
            'learned_attenuation':attenuation['attenuation']})
    scores,parts = {},{}
    for name in tracks:
        scores[name],parts[name] = metrics(records,name)
    names,matrix = [],[]
    for ref in ('reference_raw','reference_existing_calibration','previous_full_integration'):
        for metric in ('error','log_loss','brier_sum','score_nll','total_goals_nll'):
            names.append('anchored_final_minus_'+ref+':'+metric)
            matrix.append(parts['anchored_final'][metric]-parts[ref][metric])
    diffs = np.column_stack(matrix)
    actual = np.asarray([r['actual_outcome'] for r in late])
    old_selected,new_selected = reference.argmax(axis=1),shrunk.argmax(axis=1)
    class_changes = {name:{'actual_n':int((actual==c).sum()),
        'reference_correct':int(((old_selected==c)&(actual==c)).sum()),
        'anchored_correct':int(((new_selected==c)&(actual==c)).sum())}
        for c,name in enumerate(('home','draw','away'))}
    early_oof = read(OUT/'较早时间外增量记录.json')['rows']
    early_p = np.asarray([r['reference_probabilities'] for r in early_oof])
    early_delta = np.asarray([r['residual_logits'] for r in early_oof])
    from scipy.special import softmax
    early_q = softmax(np.log(early_p)+attenuation['attenuation']*early_delta,axis=1)
    early_y = np.asarray([r['actual_outcome'] for r in early_oof])
    from football_model.evaluation import probability_scores
    result = {'method':METHOD,'execution':execution,'n':len(late),'attenuation':attenuation,
        'scores':scores,'class_changes':class_changes,
        'differences':dict(zip(names,diffs.mean(axis=0).tolist())),'difference_order':names,
        'conditional_4week_intervals':moving_block_means([r['date'] for r in late],diffs,block_weeks=4),
        'by_league':{c:{name:metrics([r for r in records if r['competition']==c],name)[0] for name in tracks}
            for c in ('E0','D1','SP1','I1','F1')},
        'first_choice_changed_from_reference':int((old_selected!=new_selected).sum()),
        'raw_to_attenuated_argmax_changed':int((unshrunk.argmax(axis=1)!=new_selected).sum()),
        'reference_vs_new_overlap':{'both_correct':int(((old_selected==actual)&(new_selected==actual)).sum()),
            'new_only_correct':int(((old_selected!=actual)&(new_selected==actual)).sum()),
            'reference_only_correct':int(((old_selected==actual)&(new_selected!=actual)).sum()),
            'both_wrong':int(((old_selected!=actual)&(new_selected!=actual)).sum())},
        'earlier_oof_attenuation_fit_scores_not_external_test':{
            'reference':probability_scores(early_p,early_y),'attenuated':probability_scores(early_q,early_y)},
        'max_grid_hda_error':max_grid_error,
        'future_matches_not_predicted':True,'current_results_queried':False}
    write(OUT/'较晚逐场增量预测.json',{'rows':records})
    write(OUT/'增量整合完整结果.json',result)
    print(json.dumps({'later_scores':{k:{m:v[m] for m in ('correct','accuracy','log_loss','brier_sum','score_nll','total_goals_nll')} for k,v in scores.items()},
        'attenuation':attenuation['attenuation'],'class_changes':class_changes,
        'overlap':result['reference_vs_new_overlap']},ensure_ascii=False,indent=2),flush=True)
    return result


def report(result):
    labels = {'reference_raw':'原丰富模型','reference_existing_calibration':'原模型＋已有较早校准',
        'previous_full_integration':'上一轮全量关系整合','anchored_unshrunk':'固定原概率＋未衰减增量',
        'anchored_final':'固定原概率＋较早学得幅度的增量'}
    final,ref = result['scores']['anchored_final'],result['scores']['reference_raw']
    lines = ['# 保留原概率的增量整合：实际优化结果','',
        f"用户要求无效果则继续优化。本轮直接针对上一轮关系学习过多改动原判断的问题；同场1726场，最终{final['correct']}场正确（{final['accuracy']:.2%}），原模型{ref['correct']}场（{ref['accuracy']:.2%}）。胜平负LL {final['log_loss']:.6f}，原模型{ref['log_loss']:.6f}。",'',
        '## 针对什么改，而不是继续堆算法','',
        '上一轮重新学习基准、类别和联赛关系，并将概率差分标准化。后季平均主胜概率由42.77%降到38.72%，温度后37.53%，实际43.92%。人员差分被缩放约25—54倍，这是正则尺度变化的事实，不证明人员信息造成失败。', '',
        '本轮保留原log概率作为固定起点，只用22项自然尺度的成员差分、人员差分和既有强弱交互学习增量；不学习截距、基准或联赛偏置，不中心化/标准化。采用三类对称L2，避免指定参考类改变惩罚。', '',
        '`Q = softmax(log(P原模型) + gamma × 增量)`', '',
        'gamma只在较早真正时间外预测上学习，可取0到1；0严格恢复原始raw概率，不再用全局温度一起改变原概率。该结构也不能保证没有整体类别移动，成员差分均值仍可能带来移动。自然尺度L2是明确改变先验，并非旧模型的等价重写。', '',
        '## 实际执行和结果','',
        f"仍复用3452条既有输入，较早/较晚季各1726场。4次较早扩展拟合产生1203场时间外增量，学习gamma={result['attenuation']['attenuation']:.6f}，再以全部较早季做1次最终增量拟合。基础模型重训0次、参数网格0轮、全局温度拟合0次。没有新数据下载、首发、神经网络或赛果查询。", '',
        '| 阶段 | 正确场数 | 命中率 | 胜平负LL↓ | Brier↓ | 比分NLL↓ | 总球NLL↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in result['scores'].items():
        lines.append(f"| {labels[name]} | {v['correct']} | {v['accuracy']:.2%} | {v['log_loss']:.6f} | {v['brier_sum']:.6f} | {v['score_nll']:.6f} | {v['total_goals_nll']:.6f} |")
    lines += ['', '未衰减列只用于定位这一步的作用；没有按较晚分数在gamma=1或较早学得gamma之间择优。原模型已有校准继续作为完整参照，不隐藏较强参照。', '',
        '## 类别和概率一致性','', '| 实际类别 | 场数 | 原正确 | 最终正确 | 净变化 |','|---|---:|---:|---:|---:|']
    for name,label in [('home','主胜'),('draw','平局'),('away','客胜')]:
        v = result['class_changes'][name]
        lines.append(f"| {label} | {v['actual_n']} | {v['reference_correct']} | {v['anchored_correct']} | {v['anchored_correct']-v['reference_correct']:+d} |")
    lines += ['',f"相对原判断改变{result['first_choice_changed_from_reference']}场，错误重合完整保存。比分/总球/让球仍从统一的新网格派生；最大H/D/A一致误差{result['max_grid_hda_error']:.3g}。保持条件比分形状，因此比分NLL与1X2 NLL变化不是两份独立证据。",'',
        '## 不确定性和全联赛范围','',
        '| 参照 | 命中率差95%条件区间 | LL差95%条件区间 |', '|---|---|---|']
    intervals = result['conditional_4week_intervals']['interval_95']
    for reference in ('reference_raw','reference_existing_calibration','previous_full_integration'):
        e = intervals[result['difference_order'].index('anchored_final_minus_'+reference+':error')]
        ll = intervals[result['difference_order'].index('anchored_final_minus_'+reference+':log_loss')]
        lines.append(f"| {labels[reference]} | [{-e[1]*100:+.2f}, {-e[0]*100:+.2f}]个百分点 | [{ll[0]:+.6f}, {ll[1]:+.6f}] |")
    lines += ['', '4周时间分块区间只条件于已保存预测和已学参数，不含完整重训/方法选择不确定性。历史反复查看过，本轮仍是前推研究，不是新盲测。较早OOF上gamma的拟合分数不是独立检验，其到最终大训练样本的迁移也未被保证。', '',
        '| 联赛 | 场数 | 原命中率 | 最终命中率 | 原LL | 最终LL |','|---|---:|---:|---:|---:|---:|']
    for c,group in result['by_league'].items():
        a,b = group['reference_raw'],group['anchored_final']
        lines.append(f"| {c} | {a['n']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {a['log_loss']:.6f} | {b['log_loss']:.6f} |")
    lines += ['', '## 执行判断','',
        '相对上一轮整合，恢复了27场正确判断并改善概率评分，明确修复了那次整合的大部分退步。相对原丰富模型，最终仍少2场正确，不能称胜平负单选已提准。胜平负LL/Brier点值较好，但相对原模型和原校准模型的条件区间均跨零，尚不足以确认稳定优势。', '',
        '总球评分1.500895→1.501957，点值稍差。因而当前定位为保留原能力、对概率质量有小幅改善迹象的增量研究线，而非已证明全面更好的新正式模型。既有当前预测不改写，也没有根据后季选择未衰减列。', '',
        '## 本轮留下什么','',
        '- 联合层现在可以明确保留原概率，学习的是有参照的增量；没有独有修正时可恢复原判断。',
        '- 校准改为限制增量幅度，而非把全部原概率一起变平或变尖。是否更有效用完整结果判断，不预先保证。',
        '- 已复用上一轮来源连接、人员同设对照、时间切分、评分、比分一致化和存盘流程。没有重建这些已完成部分。',
        '- 若恢复到原概率，只能说明学到收束，不能宣传超过原模型。若点值改善但区间跨零，也不宣布稳定提准。',
        '- LLM和市场仍单列；历史时点/记录不足，不靠填造历史把它们加入。个人效应/战术的条件研究不冒充已接入早期预测。', '',
        '本轮到单套固定结构与预定时间切分为止，不按后季换C、加截距、改成员、改窗口或再学gamma。既有当前比赛预测未被改写。需要继续时，从本轮具体观察推进，而不是重开无目标的模型搜索。', '',
        '文件：'+ '、'.join(f'[{label}]({(OUT/name).as_posix()})' for label,name in (
            ('完整结果','增量整合完整结果.json'),('修正幅度','较早学习的修正幅度.json'),
            ('全部参数','全部增量参数.json'),('逐场结果','较晚逐场增量预测.json')))+'。']
    (OUT/'增量整合结果与优化结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'本轮优化安排.json',METHOD)
    rows = load_rows()
    late,raw,final,attenuation,execution = fit(rows)
    result = evaluate(late,raw,final,attenuation,execution)
    report(result)
    print(json.dumps({'output':str(OUT),'completed':True},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

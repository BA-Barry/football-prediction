"""One integrated use of existing dated forecasts, without base-model refits.

Learn a fixed low-dimensional logistic relation on earlier OOS forecasts,
calibrate it on expanding-window meta-OOS forecasts, then reconcile the
later H/D/A forecast with the existing rich score distribution.
"""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import time

import joblib
import numpy as np

from football_model.evaluation import probability_scores, calibration_bins
from football_model.forecast_diagnostics import moving_block_means
from football_model.integrated_stack import IntegratedProbabilityStack, CONFIGURATION, FEATURE_NAMES
from football_model.prematch_calibration import fit_calibrator, apply_calibrator
from football_model.probability import score_grid, summarize
from football_model.score_reconciliation import reconcile_score_probabilities


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/现有组件整合_20261026'
FINAL_DATE = date(2025, 8, 1)
EARLY, LATE = '2024-25', '2025-26'
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away',
           'actual_score', 'actual_outcome', 'forecast_before_date', 'horizon_days')
SOURCES = {
    'base': ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/lead_3/逐场完整预测.json',
    'gap': ROOT/'outputs/GAP胜平负研究_20261025/逐场胜平负概率.json',
    'kickscore': ROOT/'outputs/动态胜平负接入_20261025/forecasts.json',
    'rating': ROOT/'outputs/多路评级联合接入_20261026/逐场完整胜平负预测.json',
    'status': ROOT/'outputs/FPL赛前可用性固定对照_20261026T090728310203Z/逐场完整预测.json',
    'usage': ROOT/'outputs/FPL可用性变化参与固定对照_20261026T094122995393Z/逐场完整预测.json',
    'old_temperature': ROOT/'outputs/丰富模型单温度校准_20261026T045106381635Z/提前3日_温度参数.json',
}
METHOD = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'purpose': 'Integrate existing learned components into one coherent 1X2 and full-score candidate',
    'learner': 'one fixed StandardScaler + L2 multinomial LogisticRegression',
    'configuration': CONFIGURATION, 'features': list(FEATURE_NAMES),
    'sources': {k: str(v) for k,v in SOURCES.items()},
    'base_model_refits': 0, 'parameter_searches': 0,
    'training_period': EARLY, 'evaluation_period': LATE,
    'meta_cutoff': FINAL_DATE.isoformat(), 'horizon_days': 3,
    'meta_calibration': 'earlier season: 90 calendar-day label warmup, expanding 60-day publication windows; temperature uses only meta-OOS probabilities',
    'final_fit': 'all earlier-season OOS member predictions; parameters fixed before the later season',
    'raw_member_inputs': 'No already same-season calibrated XGBoost or apparent calibration-fit clean_blend input',
    'group_expression': 'base log odds, six member log-odds contrasts, status-control and usage-status contrasts, coverage, league, three balance interactions',
    'unknown': 'non-EPL personnel residuals are missing branches with explicit availability; never invented probability forecasts',
    'score_rule': 'keep existing rich P(score|H/D/A), scale regions to final calibrated H/D/A, recompute all means and markets',
    'primary_effect': 'later integrated calibrated minus original earlier-calibrated rich: 1X2 accuracy plus LogLoss/Brier',
    'secondary': ['full-score NLL', 'total-goals 0/1/2/3/4+ NLL', 'handicap NLL', 'draw recall', 'all leagues', 'original raw reference'],
    'stop': 'one fixed integration, planned chronological meta fits and one temperature; no later-outcome-driven retuning, new inputs or extra variants',
    'scope': 'previously examined historical seasons, date-level D-3 research; not an untouched test or a new current-match prediction',
    'not_quantitatively_included': {
        'LLM': 'No sufficiently complete uncontaminated earlier/later historical LLM probability series; preserve independent route and evidence review',
        'market': 'No verified D-3 historical quote path; preserve separate market probability',
        'conditional_players': 'Actual target minutes/known XI unavailable at early prediction; reuse available status/participation, not old conditional effects',
        'neural_RL_full_simulation': 'not trained or required for this integration',
    },
    'current_predictions_replaced': False, 'current_results_queried': False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def index(rows):
    rows = list(rows)
    mapped = {r['match_id']: r for r in rows}
    if len(mapped) != len(rows):
        raise ValueError('Duplicate source forecast identity')
    return mapped


def matched(reference, item):
    if any(reference[k] != item[k] for k in CONTEXT):
        raise ValueError('Member fixture, result or publication mismatch: '+reference['match_id'])
    if not item['fit_before_date'] <= item['forecast_before_date'] < item['date']:
        raise ValueError('Future member fit or invalid source chronology')


def prepare():
    original = read(SOURCES['base'])
    members = {
        'rich_performance': index(original['fresh_performance']),
        'rich_allcomp': index(original['fresh_allcomp']),
        'goal': index(original['goal']), 'xg': index(original['xg_raw']),
        'gap': index(read(SOURCES['gap'])['rows']),
        'kickscore': index(r for r in read(SOURCES['kickscore'])['tracks']['dynamic'] if r['horizon_days']==3),
        'rating_joint': index(read(SOURCES['rating'])['rich_joint']),
    }
    status, usage = read(SOURCES['status']), read(SOURCES['usage'])
    personnel = {
        'availability_control': index(status['epl_control']),
        'availability_status': index(status['epl_availability']),
        'availability_usage': index(usage['availability_exposure']),
    }
    other_control = index(usage['epl_control'])
    ref = members['rich_performance']
    if any(set(group) != set(ref) for group in members.values()):
        raise ValueError('Keep complete identical source cohorts; do not intersect away failures')
    epl_ids = {key for key,r in ref.items() if r['competition']=='E0'}
    if any(set(group) != epl_ids for group in personnel.values()) or set(other_control) != epl_ids:
        raise ValueError('Personnel routes must exactly cover the EPL subset')
    rows, fit_differences = [], Counter()
    for key,r in sorted(ref.items(), key=lambda kv:(kv[1]['date'],kv[0])):
        if r['horizon_days'] != 3 or date.fromisoformat(r['date'])-timedelta(days=3) != date.fromisoformat(r['forecast_before_date']):
            raise ValueError('Every target must retain the archived D-3 publication')
        actual = r['actual_score']
        expected = 0 if actual[0]>actual[1] else 1 if actual[0]==actual[1] else 2
        if expected != r['actual_outcome']:
            raise ValueError('Full score and ordered outcome disagree')
        fits = {}
        for name,group in members.items():
            matched(r,group[key])
            fits[name] = group[key]['fit_before_date']
            fit_differences[name] += int(fits[name] != r['fit_before_date'])
        for group in personnel.values():
            if key in group:
                matched(r,group[key])
        if key in other_control:
            matched(r,other_control[key])
            if not np.array_equal(other_control[key]['probabilities'],personnel['availability_control'][key]['probabilities']):
                raise ValueError('Personnel contrasts require identical own same-setting control')
        rows.append({**{k:r[k] for k in CONTEXT},
            'probabilities_by_route': {name: group[key]['probabilities'] for name,group in members.items()},
            **{name: group[key]['probabilities'] if key in group else None for name,group in personnel.items()},
            'original_rates':r['rates'], 'route_fit_before_dates':fits})
    with gzip.open(OUT/'整合赛前输入.json.gz','wt',encoding='utf-8') as f:
        json.dump({'rows':rows},f,ensure_ascii=False,allow_nan=False)
    coverage = {'n_rows':len(rows),'by_period':dict(Counter(r['period'] for r in rows)),
        'by_league':dict(Counter(r['competition'] for r in rows)),
        'personnel_observed':sum(r['availability_status'] is not None for r in rows),
        'parameter_dates_different_from_rich':dict(fit_differences),
        'timing_note':'Same publication day and fixture; KickScore updates on a different valid earlier schedule, not the rich 28-day schedule',
        'member_names':list(members),'feature_n':len(FEATURE_NAMES),
        'excluded_target_rows':0,'source_probabilities_rewritten':False}
    write(OUT/'输入与覆盖.json',coverage)
    return rows


def fit_meta(rows):
    early = [r for r in rows if r['period']==EARLY]
    late = [r for r in rows if r['period']==LATE]
    if not early or not late or any(r['date']>=FINAL_DATE.isoformat() for r in early) or any(r['forecast_before_date']<FINAL_DATE.isoformat() for r in late):
        raise ValueError('Keep complete, separated earlier and later seasons')
    start = min(date.fromisoformat(r['date']) for r in early)+timedelta(days=90)
    last = max(date.fromisoformat(r['forecast_before_date']) for r in early)
    windows, oof, parameters = [], [], {}
    model_dir = OUT/'models'
    model_dir.mkdir(exist_ok=True)
    cutoff = start
    while cutoff <= last:
        end = cutoff+timedelta(days=60)
        training = [r for r in early if r['date']<cutoff.isoformat()]
        targets = [r for r in early if cutoff.isoformat()<=r['forecast_before_date']<end.isoformat()]
        if targets:
            t = time.perf_counter()
            stack = IntegratedProbabilityStack().fit(training,cutoff)
            p = stack.predict_many(targets)
            name = 'meta_'+cutoff.isoformat()
            joblib.dump(stack,model_dir/(name+'.joblib'),compress=3)
            parameters[name] = stack.parameters()
            oof.extend({**{k:r[k] for k in CONTEXT},'probabilities':v.tolist(),
                        'fit_before_date':cutoff.isoformat()}
                       for r,v in zip(targets,p,strict=True))
            windows.append({'cutoff':cutoff.isoformat(),'publication_end_exclusive':end.isoformat(),
                'training_n':len(training),'held_n':len(targets),'seconds':time.perf_counter()-t})
            print(json.dumps({'meta_oof_window':windows[-1]},ensure_ascii=False),flush=True)
        cutoff = end
    oof.sort(key=lambda r:(r['date'],r['match_id']))
    if len({r['match_id'] for r in oof})!=len(oof):
        raise ValueError('Meta OOS windows must not overlap')
    temperature = fit_calibrator(oof,FINAL_DATE,'temperature')
    final = IntegratedProbabilityStack().fit(early,FINAL_DATE)
    parameters['final'] = final.parameters()
    joblib.dump(final,model_dir/'integrated_stack_2025-08-01.joblib',compress=3)
    # Explicit executable check: later labels are not prediction features.
    q = final.predict_many(late)
    altered = [{**r,'actual_outcome':(r['actual_outcome']+1)%3,'actual_score':[99,98]} for r in late]
    if not np.array_equal(q,final.predict_many(altered)):
        raise ValueError('Prediction must be invariant to withheld later labels/scores')
    restored = joblib.load(model_dir/'integrated_stack_2025-08-01.joblib')
    if not np.array_equal(q,restored.predict_many(late)):
        raise ValueError('Saved integration must reproduce probabilities exactly')
    calibrated = apply_calibrator(q,temperature)
    write(OUT/'全部整合参数.json',parameters)
    write(OUT/'整合温度参数.json',temperature)
    write(OUT/'较早整合时间外预测.json',{'rows':oof,'windows':windows,
          'uncovered_warmup_n':len(early)-len(oof),'note':'No in-sample final-stack forecasts used to fit this temperature'})
    fits = {'base_fits':0,'meta_oof_fits':len(windows),'final_meta_fits':1,'temperature_fits':1,
        'reused_reference_temperature':str(SOURCES['old_temperature']),'parameter_searches':0,
        'later_labels_invariance':True,'saved_restore_exact':True}
    write(OUT/'执行记录.json',fits)
    print(json.dumps({'fit_summary':fits,'temperature':temperature['parameters']},ensure_ascii=False),flush=True)
    return late, q, calibrated, fits


def metrics(records,key):
    p = np.asarray([r['probabilities'][key] for r in records])
    y = np.asarray([r['actual_outcome'] for r in records])
    selected = p.argmax(axis=1)
    confusion = np.zeros((3,3),dtype=int)
    np.add.at(confusion,(y,selected),1)
    base = probability_scores(p,y)
    base.update({'correct':int((selected==y).sum()),
        'mean_probabilities':p.mean(axis=0).tolist(),
        'confusion_actual_rows_predicted_columns':confusion.tolist(),
        'selected_class_n':np.bincount(selected,minlength=3).tolist(),
        'class_recall':[float(confusion[i,i]/confusion[i].sum()) for i in range(3)],
        'draw_precision':float(confusion[1,1]/confusion[:,1].sum()) if confusion[:,1].sum() else None,
        'calibration_10_bins':calibration_bins(p,y,bins=10)})
    per = {'log_loss':-np.log(p[np.arange(len(y)),y]),
        'brier_sum':np.square(p-np.eye(3)[y]).sum(axis=1),
        'error':(selected!=y).astype(float)}
    for name in ('score_nll','total_goals_nll','over_2_5_brier','home_minus_one_nll','away_minus_two_nll'):
        values = np.asarray([r['market_losses'][key][name] for r in records])
        base[name] = float(values.mean())
        per[name] = values
    return base,per


def derive_grid_losses(grid,actual):
    h,a = np.indices(grid.grid.shape)
    total = actual[0]+actual[1]
    observed_total = grid.grid[h+a==total].sum() if total<4 else grid.grid[h+a>=4].sum()
    actual_home_handicap = np.sign(actual[0]-actual[1]-1)
    actual_away_handicap = np.sign(actual[1]-actual[0]-2)
    return {'score_nll':float(-np.log(grid.grid[actual[0],actual[1]])),
        'total_goals_nll':float(-np.log(observed_total)),
        'over_2_5_brier':float((grid.grid[h+a>=3].sum()-(total>=3))**2),
        'home_minus_one_nll':float(-np.log(grid.grid[np.sign(h-a-1)==actual_home_handicap].sum())),
        'away_minus_two_nll':float(-np.log(grid.grid[np.sign(a-h-2)==actual_away_handicap].sum()))}


def evaluate(late,raw,calibrated,fits):
    old_temp = read(SOURCES['old_temperature'])
    if old_temp['before_date']!=FINAL_DATE.isoformat() or old_temp['n_matches']!=1726:
        raise ValueError('Use the existing correctly dated reference temperature')
    rich = np.asarray([r['probabilities_by_route']['rich_performance'] for r in late])
    rich_cal = apply_calibrator(rich,old_temp)
    tracks = {'original_rich':rich,'original_rich_calibrated':rich_cal,
              'integrated_raw':raw,'integrated_calibrated':calibrated}
    records, max_error = [],0.
    for i,r in enumerate(late):
        original_grid = score_grid(*r['original_rates'])
        if not np.allclose(original_grid.home_draw_away,rich[i],rtol=0,atol=1e-10):
            raise ValueError('Reconstruct only the identical archived rich score distribution')
        losses, summaries = {},{}
        for key,p in tracks.items():
            grid = original_grid if key=='original_rich' else reconcile_score_probabilities(original_grid,p[i])
            actual_probs = np.asarray(grid.home_draw_away)
            error = float(np.abs(actual_probs-p[i]).max())
            max_error = max(max_error,error)
            if error>1e-10:
                raise ValueError('All final score markets must agree with declared H/D/A')
            losses[key] = derive_grid_losses(grid,r['actual_score'])
            if key in ('original_rich','integrated_calibrated'):
                summaries[key] = summarize(grid)
        records.append({**{k:r[k] for k in CONTEXT},
            'probabilities':{key:p[i].tolist() for key,p in tracks.items()},
            'market_losses':losses,'final_markets':summaries,
            'source_probabilities':r['probabilities_by_route'],
            'personnel_probability_available':r['availability_status'] is not None,
            'integration_before_date':FINAL_DATE.isoformat(),
            'new_first_choice':int(calibrated[i].argmax()),
            'original_first_choice':int(rich[i].argmax())})
    scores,components = {},{}
    for key in tracks:
        scores[key],components[key] = metrics(records,key)
    columns,deltas = [],[]
    for reference in ('original_rich','original_rich_calibrated'):
        for metric in ('error','log_loss','brier_sum','score_nll','total_goals_nll','home_minus_one_nll','away_minus_two_nll'):
            columns.append('integrated_calibrated_minus_'+reference+':'+metric)
            deltas.append(components['integrated_calibrated'][metric]-components[reference][metric])
    matrix = np.column_stack(deltas)
    original_right = rich.argmax(axis=1)==np.asarray([r['actual_outcome'] for r in late])
    joint_right = calibrated.argmax(axis=1)==np.asarray([r['actual_outcome'] for r in late])
    result = {'method':METHOD,'execution':fits,'n':len(late),'scores':scores,
        'differences':dict(zip(columns,matrix.mean(axis=0).tolist())),
        'difference_order':columns,
        'conditional_4week_intervals':moving_block_means([r['date'] for r in late],matrix,block_weeks=4),
        'by_league':{c:{key:metrics([r for r in records if r['competition']==c],key)[0] for key in tracks} for c in COMPS},
        'changes':{'first_choice_changed_n':int((rich.argmax(axis=1)!=calibrated.argmax(axis=1)).sum()),
            'both_correct':int((original_right & joint_right).sum()),
            'new_only_correct':int((~original_right & joint_right).sum()),
            'original_only_correct':int((original_right & ~joint_right).sum()),
            'both_wrong':int((~original_right & ~joint_right).sum()),
            'temperature_argmax_changed_n':int((raw.argmax(axis=1)!=calibrated.argmax(axis=1)).sum())},
        'max_reconciled_hda_error':max_error,
        'note':'Intervals condition on saved predictions and fitted meta/calibration; no full training/model-selection uncertainty. All seasons previously viewed.',
        'current_forecasts_replaced':False,'current_match_results_queried':False}
    write(OUT/'较晚逐场联合预测.json',{'rows':records})
    write(OUT/'整合完整结果.json',result)
    print(json.dumps({'later_scores':{k:{m:v[m] for m in ('n','correct','accuracy','log_loss','brier_sum','score_nll','total_goals_nll')} for k,v in scores.items()},'changes':result['changes'],'max_reconciled_error':max_error},ensure_ascii=False,indent=2),flush=True)
    return result


def report(result):
    labels = {'original_rich':'原丰富表现','original_rich_calibrated':'原丰富表现＋已有较早校准',
              'integrated_raw':'整合关系（校准前）','integrated_calibrated':'整合关系＋时间外校准＋统一比分'}
    old = result['scores']['original_rich']
    integrated = result['scores']['integrated_calibrated']
    lines = ['# 现有组件整合：实际执行与整体效果','',
        f"本轮已完成一次真实整合，整体没有改善：较晚同场{result['n']}场，单选{old['correct']}→{integrated['correct']}场（{old['accuracy']:.2%}→{integrated['accuracy']:.2%}），胜平负LogLoss {old['log_loss']:.6f}→{integrated['log_loss']:.6f}。不据此覆盖当前预测，也不据此否定组件本身。",'',
        '按用户要求，先把当前可用积累整合为一套方案：已有概率收束、正则化学习它们的共同关系、严格时间外校准，再形成一致比分分布。没有重新搜索模型或重训基础成员，也没有把缺历史的LLM、赔率或条件球员强度硬填进来。','',
        '## 整合了什么','',
        '| 环节 | 本轮实际使用 |','|---|---|',
        '| 基础信息 | 121丰富表现、161含全赛事、进球、xG、GAP、KickScore动态、多路评级丰富XGBoost |',
        '| 人员 | 135状态相对同设英超对照、143参与相对135状态；其他联赛明确无覆盖 |',
        '| 关系学习 | 29项概率对比、赛事及少量强弱交互；固定L2多项逻辑回归，训练期标准化 |',
        '| 校准 | 2024/25内部按时间扩展得到联合模型的样本外预测，只用这些学习一个共享温度 |',
        '| 比分与市场 | 既有121比分作为条件形状，按最终H/D/A区域缩放；均值、总球和让球全部重算 |',
        '| 继续单列 | LLM和市场：历史系列/时点不足，本轮没有伪造可训练历史；保留独立输入角色 |','',
        '这些成员共享大量信息。差分表示和正则用于共同学习，不能把每个成员或系数称作独立证据/独立因果贡献。人员描述也不是完整个人能力模型。', '',
        '## 时间与真实执行','',
        '各成员直接使用较早时点已有的原始概率，避免把同年全年标签校准后的概率再次冒充样本外成员。2024/25学习联合关系；内部初始90天标签热身，之后每60天扩展训练，按预测发布日期划分目标。2025-08-01最终拟合固定到2025/26，完整1726场全部评价。KickScore更新频率与其他成员不同，但信息截止日期一致，保留其实际日期。','',
        f"实际：基础模型重训0次；时间外联合拟合{result['execution']['meta_oof_fits']}次；最终联合拟合1次；温度拟合1次；参数网格0轮。没有新增依赖、目标首发、神经网络或当前赛果查询。",'',
        '## 同一批较晚1726场的整体变化','',
        '| 阶段 | 单选命中 | 命中率 | 胜平负LogLoss↓ | Brier↓ | 比分NLL↓ | 总球NLL↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,v in result['scores'].items():
        lines.append(f"| {labels[key]} | {v['correct']}/{v['n']} | {v['accuracy']:.2%} | {v['log_loss']:.6f} | {v['brier_sum']:.6f} | {v['score_nll']:.6f} | {v['total_goals_nll']:.6f} |")
    lines += ['', '原模型校准后的比分/总球列，也是按其校准H/D/A重分配原比分后新算；旧报告只有1X2校准，这里没有冒称旧比分分布原先已改变。总球是0/1/2/3/4+五档评分，命中率只取主/平/客概率最高一项，未筛选容易比赛。','',
        '## 各联赛完整保留','',
        '| 联赛 | 场数 | 原命中率 | 整合命中率 | 原LL | 整合LL |',
        '|---|---:|---:|---:|---:|---:|']
    for c,group in result['by_league'].items():
        a,b = group['original_rich'],group['integrated_calibrated']
        lines.append(f"| {c} | {a['n']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {a['log_loss']:.6f} | {b['log_loss']:.6f} |")
    lines += ['', '## 变化与不确定性','']
    change = result['changes']
    lines.append(f"相对原丰富表现，最高选项改变{change['first_choice_changed_n']}场；{change['both_correct']}场两者都对、{change['new_only_correct']}场仅整合对、{change['original_only_correct']}场仅原判断对、{change['both_wrong']}场都错。数字只描述已完成预测，没有事后挑正确成员。")
    lines += ['', '| 相对对象 | 命中率差95%条件区间 | LL差95%条件区间 | Brier差95%条件区间 |',
              '|---|---|---|---|']
    intervals = result['conditional_4week_intervals']['interval_95']
    for ref in ('original_rich','original_rich_calibrated'):
        entries = []
        for metric in ('error','log_loss','brier_sum'):
            name = 'integrated_calibrated_minus_'+ref+':'+metric
            lo,hi = intervals[result['difference_order'].index(name)]
            entries.append(f'[{(-hi)*100:+.2f}, {(-lo)*100:+.2f}]个百分点' if metric=='error' else f'[{lo:+.6f}, {hi:+.6f}]')
        lines.append('| '+labels[ref]+' | '+' | '.join(entries)+' |')
    lines += ['', '区间为既有预测和已学参数条件下的4周时间分块描述，不包含完整重训、方法选择及资料修订的不确定性。2024/25和2025/26都已多轮查看，仍是时间前推研究，不能称新盲测。早期小样本联合模型与最终全季联合模型的校准迁移也存在不确定性。','',
        '## 平局与其他市场','', '| 项目 | 原丰富表现 | 整合后 |','|---|---:|---:|']
    a,b = result['scores']['original_rich'],result['scores']['integrated_calibrated']
    lines.append(f"| 选平场数 | {a['selected_class_n'][1]} | {b['selected_class_n'][1]} |")
    lines.append(f"| 实际平局召回 | {a['class_recall'][1]:.2%} | {b['class_recall'][1]:.2%} |")
    for k,lab in [('home_minus_one_nll','主−1竞彩NLL'),('away_minus_two_nll','客−2竞彩NLL'),('over_2_5_brier','总球大2.5 Brier')]:
        lines.append(f'| {lab} | {a[k]:.6f} | {b[k]:.6f} |')
    lines += ['', '比分区域缩放保留P(比分|主/平/客)，是一项明确的一致化假设；不会凭空学习区域内部的1:1与2:2结构。它会改变原主客进球均值和总球，因此这些市场必须一起检视。已逐场确认最终网格与联合H/D/A一致。','',
        '## 本轮收束','',
        '到这一套整合、完整后季与既定任务检查为止。没有根据较晚成绩换C、删成员、改窗口、加联赛专属参数或续做变体。结果用于理解已有组件如何共同起作用，以及哪里仍缺可迁移关系；不会因某个整体分数就丢弃已学组件。当前比赛的既有模型、LLM及联合记录保持原样。','',
        '需要结合这里的实际指标作执行判断；不能把多项指标混成后验挑选的单一“成功”标签。神经、个人战术及后续核心调整仍按用户偏好同步。','',
        '已保存结果的零训练定位显示：主胜正确判断602→509（−93），平局0→30、客胜299→333（合计＋64），是命中净下降29场的直接数量分解。这不证明某一成员造成了损失。温度在较早拟合资料上改善，在较晚季没有延续；没有据此重选温度或回退报告。', '',
        '留下的可复用成果是“来源明确的概率收束 → 正则联合关系 → 时间外校准 → 比分一致化”的完整可执行流程。各步骤可独立调用；一致化保持条件比分形状，比分NLL变化等于1X2 NLL变化，两者不能当作独立的重复证据。现阶段仍需识别如何保存已有强弱关系、利用真正独有信息，而不是再增加模型数量。','',
        '文件：'+ '、'.join(f'[{label}]({(OUT/name).as_posix()})' for label,name in (
            ('完整结果','整合完整结果.json'),('全部参数','全部整合参数.json'),
            ('逐场概率与市场','较晚逐场联合预测.json'),('较早时间外记录','较早整合时间外预测.json'),
            ('变化定位与可复用结论','变化定位与可复用结论.md')))+'。']
    (OUT/'整合结果与所学.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'整合安排.json',METHOD)
    rows = prepare()
    late,raw,calibrated,fits = fit_meta(rows)
    result = evaluate(late,raw,calibrated,fits)
    report(result)
    print(json.dumps({'output':str(OUT),'completed':True},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

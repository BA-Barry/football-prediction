"""Execute one fixed conditional direction study; reuse each saved fitted window."""
import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import joblib
import numpy as np

from football_model.conditional_direction import ConditionalDirectionLearner
from football_model.persistent_rich_integration import plan_windows
from football_model.persistent_state_features import STATE_FEATURE_NAMES
from football_model.probability import score_grid, summarize
from football_model.score_reconciliation import reconcile_score_probabilities
from football_model.forecast_diagnostics import moving_block_means
from run_persistent_rich_integration_20261028 import loss_arrays, metrics


ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/条件主客方向研究_20261028'
RICH = ROOT/'outputs/持续状态与丰富信息接入_20261028'
DRAW = ROOT/'outputs/历史比赛模式与平局分支_20261027'
TRACKS = ('original', 'existing_draw', 'state_rich', 'conditional')
LABELS = ('主胜', '平局', '客胜')
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away', 'actual_score',
           'actual_outcome', 'fit_before_date', 'forecast_before_date', 'feature_before_date', 'horizon_days')


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def save(path, data):
    with (gzip.open(path, 'wt', encoding='utf-8') if path.suffix == '.gz'
          else path.open('w', encoding='utf-8')) as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False,
                  indent=None if path.suffix == '.gz' else 2)


def index(rows):
    result = {r['match_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate fixture IDs')
    return result


def inputs():
    rows = read(RICH/'联合输入_待训练.json.gz')['rows']
    originals = index(read(RICH/'接入逐场概率.json.gz')['records'])
    draws = index(read(DRAW/'逐场概率与纠错.json')['records'])
    parameters = read(DRAW/'全部分类模型参数.json')
    if set(originals) != set(draws) or len(originals) != 3452:
        raise ValueError('Retain the identical complete 3452 fixture cohort')
    windows = plan_windows(rows)
    by_id = index(rows)
    for window in windows:
        if parameters['extra_trees_pattern:'+window['cutoff']]['reference_match_ids'] != window['training_ids']:
            raise ValueError('Existing draw model training IDs or chronology differ')
        for key in window['target_ids']:
            r, old, d = by_id[key], originals[key], draws[key]
            if any(r[k] != old[k] or r[k] != d[k] for k in CONTEXT):
                raise ValueError('Saved target context, P/F or labels differ')
            if old['tracks']['original']['probabilities'] != d['original_probabilities']:
                raise ValueError('Original reference probabilities differ')
    if len(windows) != 22:
        raise ValueError('Use the complete existing 22 windows')
    return rows, originals, draws, windows


def projected(rows, window):
    history = [row for row in rows if row['date'] < window['cutoff']]
    if [r['match_id'] for r in sorted(history, key=lambda r:(r['date'],r['match_id']))] != window['training_ids']:
        raise ValueError('Earlier history changed')
    conditional = [r for r in history if r['actual_outcome'] != 1]
    active = [name for name in STATE_FEATURE_NAMES
              if any(r['side_features'][0][name] is not None for r in conditional)]
    inactive = set(STATE_FEATURE_NAMES)-set(active)
    names = sorted(set(rows[0]['side_features'][0])-inactive)
    def convert(r):
        return {**r, 'match_features': {name:r['side_features'][0][name] for name in names}}
    held_ids = set(window['target_ids'])
    held = sorted([r for r in rows if r['match_id'] in held_ids], key=lambda r:(r['date'],r['match_id']))
    return [convert(r) for r in history], [convert(r) for r in held], active, names


def package(grid, score):
    h,a = score
    mass = float(grid.grid[h,a])
    if mass <= 0:
        raise ValueError('Zero exact-score mass, do not smooth the loss')
    return {'probabilities':list(map(float,grid.home_draw_away)),
            'forecast':summarize(grid), 'exact_score_nll':-math.log(mass)}


def evaluate(records):
    values = {name:metrics(records,name) for name in TRACKS}
    y=np.asarray([r['actual_outcome'] for r in records])
    truth=np.eye(3)[y]
    for name in TRACKS:
        p=np.asarray([r['tracks'][name]['probabilities'] for r in records])
        values[name]['mean_losses']['rps']=float(np.square(
            np.cumsum(p,axis=1)[:,:2]-np.cumsum(truth,axis=1)[:,:2]).sum(axis=1).mean()/2)
    comparisons = {}
    new = loss_arrays(records,'conditional')[3]
    for reference in ('existing_draw','state_rich','original'):
        old = loss_arrays(records,reference)[3]
        cols = list(new)
        differences = np.column_stack([new[k]-old[k] for k in cols])
        old_top = np.asarray([r['tracks'][reference]['probabilities'] for r in records]).argmax(axis=1)
        top = np.asarray([r['tracks']['conditional']['probabilities'] for r in records]).argmax(axis=1)
        comparisons[reference] = {'difference_order':cols,
            'new_minus_reference':dict(zip(cols,differences.mean(axis=0).tolist())),
            'changed':int((top != old_top).sum()),
            'corrected':int(((old_top != y)&(top == y)).sum()),
            'damaged':int(((old_top == y)&(top != y)).sum()),
            'both_wrong_changed':int(((top!=old_top)&(top!=y)&(old_top!=y)).sum()),
            'four_week_conditional_intervals':moving_block_means(
                [r['date'] for r in records], differences, block_weeks=4)
                if reference != 'original' else None}
    return {'n':len(records),'tracks':values,'comparisons':comparisons}


def report(result):
    lines=['# 条件主客方向：完整固定研究结果','',
        '每窗只用更早非平比赛拟合q；d复用已有历史模式组件。全部3452目标保留，包括实际平局。',
        '22窗固定设置，无追加阈值、种子、权重或参数搜索。历史已多轮用于研发，不是新盲测或当前准确率。','',
        '|赛季|线路|正确/1726|H/D/A LL↓|平/非平LL↓|条件方向贡献↓|总球LL↓|比分LL↓|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for period,g in result['by_period'].items():
        for name,m in g['tracks'].items():
            l=m['mean_losses']
            lines.append(f"|{period}|{name}|{m['correct']}|{l['one_x_two']:.6f}|{l['draw_binary']:.6f}|{l['conditional_direction_weighted']:.6f}|{l['total_goals']:.6f}|{l['exact_score']:.6f}|")
    lines += ['', '|赛季|相对已有同d平局线的LL差|四周条件区间|纠正/改坏/双方错的改向|',
              '|---|---:|---|---:|']
    for period,g in result['by_period'].items():
        c=g['comparisons']['existing_draw']
        i=c['difference_order'].index('one_x_two')
        interval=c['four_week_conditional_intervals']['interval_95'][i]
        lines.append(f"|{period}|{c['new_minus_reference']['one_x_two']:+.6f}|[{interval[0]:+.6f}, {interval[1]:+.6f}]|{c['corrected']}/{c['damaged']}/{c['both_wrong_changed']}|")
    lines += ['', '原121项、已存状态、既有平局信号与预处理均保留。新q不等于首次直接胜平负；改变的是条件训练样本和树分裂任务，不能把全部差异因果归于一个因素。',
        '平局概率固定，故相对同d参照的平局损失差应仅为浮点误差，完整H/D/A损失变化来自条件方向。训练非平样本是合法条件目标；赛前不能用目标真实结果筛比赛。',
        '比分区域缩放保留v0.50的P(score|H/D/A)，新q没有学习条件比分；总球、让球和比分均值必须从新网格重算。',
        '四周区间仅条件于已保存预测，未涵盖全部拟合与多轮研发选择不确定性。不从后季挑路线或分组。',
        '**本项已到固定完整研究停止位置。保留研究资产，不自动替换当前预测，不追加参数救分。**','',
        '## 实际结论与可复用认识','',
        '本次条件头两季命中930／1726、889／1726，原121项为923、901：早季+7、晚季-12，合计少5场。相对固定同一d的已有平局组件则为+8、-9。不能以一个赛季、单选指标或很小的晚季LL改善宣布采纳。','',
        '同d参照下，新方向头的LL差早季+0.001496、晚季-0.000098，四周条件区间均跨0。晚季相对v0.50的LL改善约0.001549，其中平局部分改善0.001583、方向部分反而变差0.000034；主要收益来自已经存在的d，不能记作新方向学习的收益。','',
        '已保留的资产是可单独回放的条件学习器、22窗模型、固定d接口与完整比分统一，不是已提准的正式替代。q更分散且平均改动约3.3个百分点；这会影响概率质量，单选没变的比赛也必须计分。d固定仍可能改变三项中平局是否最大，不能把平局首选数变化解释成学到了新的平局信息。','',
        '输入时间和训练ID已经独立重建，保存模型对全部3452场重放一致；这说明本轮没有把目标放入同窗训练，不能消除这些赛季反复研发造成的选择偏差。独立核对及机制说明另存本目录。','',
        '比分相对已有平局线的变化还含v0.50条件比分形状来源差异；只有固定d的H/D/A损失分解可以在这里直接归到q。没有把新q用于10月28日比赛，也没有用昨日三场高进球结果调参。','',
        '下一步先处理应用条件：实际预测用了同日资料，保存121项树的历史资料提前3天，这一差异必须在同提前量历史输入中研究；同时继续保留新前瞻的原始分歧，累计后按单项概率而非双选覆盖评价。补充资料先列清赛程、人员与战术的已知／未知，暂不新增算法叠加或再救本轮参数。','']
    (OUT/'本轮结果与所学.md').write_text('\n'.join(lines),encoding='utf-8')


def summarize_saved():
    records=read(OUT/'逐场概率与条件方向.json.gz')['records']
    if len(index(records)) != 3452:
        raise ValueError('Incomplete final cohort')
    result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'by_period':{period:evaluate([r for r in records if r['period']==period])
                     for period in ('2024-25','2025-26')},
        'target_n':len(records), 'expected_fitted_windows':22,
        'current_forecasts_replaced':False, 'parameter_grids':0,
        'upstream_draw_refits':0, 'upstream_gas_refits':0,
        'scope':'reused development historical forward study, not blind validation'}
    save(OUT/'完整结果.json',result)
    report(result)
    with (OUT/'全部改向记录.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['match_id','date','period','actual_outcome','reference','old_top','new_top',
                *[f'{route}_{label}' for route in ('old','new') for label in ('H','D','A')]]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            new=r['tracks']['conditional']['probabilities']; nt=int(np.argmax(new))
            for reference in ('original','existing_draw','state_rich'):
                old=r['tracks'][reference]['probabilities'];ot=int(np.argmax(old))
                if nt==ot:continue
                row={key:r[key] for key in ('match_id','date','period','actual_outcome')}
                row.update(reference=reference,old_top=ot,new_top=nt)
                row.update(dict(zip(('old_H','old_D','old_A','new_H','new_D','new_A'),old+new)))
                writer.writerow(row)
    return result


def run(rows, originals, draws, windows):
    (OUT/'models').mkdir(parents=True,exist_ok=True)
    (OUT/'windows').mkdir(exist_ok=True)
    records,grids,parameters,progress=[],[],{},[]
    for window in windows:
        started=time.perf_counter();cutoff=window['cutoff']
        history,held,active,names=projected(rows,window)
        model_path=OUT/'models'/f'conditional_direction_{cutoff}.joblib'
        fitted=False
        if model_path.exists():
            model=joblib.load(model_path)
        else:
            model=ConditionalDirectionLearner().fit(history,date.fromisoformat(cutoff))
            joblib.dump(model,model_path,compress=3);fitted=True
        p=model.parameters()
        expected=[r['match_id'] for r in sorted(history,key=lambda r:(r['date'],r['match_id'])) if r['actual_outcome']!=1]
        if (p['reference_match_ids']!=expected or p['earlier_all_match_ids']!=window['training_ids']
                or p['feature_names']!=names or p['before_date']!=cutoff):
            raise ValueError('Saved conditional model training scope differs')
        parameters[cutoff]={'model':p,'active_state_features':active,'model_file':str(model_path)}
        checkpoint=OUT/'windows'/f'{cutoff}.json.gz'
        if checkpoint.exists():
            data=read(checkpoint)
            if [r['match_id'] for r in data['records']] != [r['match_id'] for r in held]:
                raise ValueError('Saved prediction window targets differ')
        else:
            qs=model.predict_many([r['match_features'] for r in held],
                                  [date.fromisoformat(r['forecast_before_date']) for r in held])
            data={'records':[],'grids':[]}
            for row,q in zip(held,qs,strict=True):
                fid=row['match_id'];old=originals[fid];drow=draws[fid];d=drow['classifier_draw_probability']
                hda=[(1-d)*q,d,(1-d)*(1-q)]
                grid=reconcile_score_probabilities(score_grid(*old['tracks']['state_rich']['rates'],rho=0),hda)
                candidate=package(grid,row['actual_score'])
                if not np.allclose(candidate['probabilities'],hda,atol=1e-12,rtol=0):
                    raise ValueError('Reconciled grid does not preserve full H/D/A')
                existing={'probabilities':drow['probabilities'],'forecast':drow['forecast'],
                          'exact_score_nll':drow['exact_score_nll']}
                data['records'].append({**{k:row[k] for k in CONTEXT},
                    'conditional_home_probability':q,'fixed_draw_probability':d,
                    'active_state_features':active,'state_history_max_date':row['state_history_max_date'],
                    'state_parameter_training_max_date':row['state_parameter_training_max_date'],
                    'tracks':{'original':old['tracks']['original'], 'existing_draw':existing,
                              'state_rich':old['tracks']['state_rich'],'conditional':candidate}})
                data['grids'].append({'match_id':fid,'probabilities':grid.grid.tolist(),
                    'axis':['home_goals','away_goals'],'source_rates':old['tracks']['state_rich']['rates']})
            save(checkpoint,data)
        records.extend(data['records']);grids.extend(data['grids'])
        progress.append({'cutoff':cutoff,'new_fit':fitted,'training_n':p['n_matches'],
                         'target_n':len(held),'active_state_n':len(active),'seconds':time.perf_counter()-started})
        save(OUT/'模型参数.json',parameters)
        save(OUT/'执行进度.json',{'windows':progress,'new_fits_this_execution':sum(r['new_fit'] for r in progress)})
        print(json.dumps({'stage':'window_complete',**progress[-1]},ensure_ascii=False),flush=True)
    records.sort(key=lambda r:(r['date'],r['match_id']))
    save(OUT/'逐场概率与条件方向.json.gz',{'records':records})
    save(OUT/'逐场统一比分网格.json.gz',{'records':grids})
    return summarize_saved()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inspect','run','summary'),default='inspect')
    args=parser.parse_args()
    if args.mode=='summary':
        result=summarize_saved()
    else:
        rows,originals,draws,windows=inputs()
        if args.mode=='inspect':
            counts=[]
            for w in windows:
                h,t,active,names=projected(rows,w)
                counts.append({'cutoff':w['cutoff'],'conditional_training_n':sum(r['actual_outcome']!=1 for r in h),
                               'target_n':len(t),'feature_n':len(names)})
            print(json.dumps({'input_n':len(rows),'targets':len(originals),'new_fits':0,'windows':counts},ensure_ascii=False))
            return
        result=run(rows,originals,draws,windows)
    print(json.dumps({'stage':'complete','by_period':{period:{name:m['correct'] for name,m in g['tracks'].items()}
        for period,g in result['by_period'].items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

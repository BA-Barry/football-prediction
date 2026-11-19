"""One existing temperature fit per horizon on saved rich-model forecasts.

Earlier 2024/25 calibration, fixed application in 2025/26. No base fits, grids,
group-specific parameters, current target scoring or forecast replacement.
"""
from datetime import date,datetime,timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.evaluation import probability_scores,calibration_bins
from football_model.forecast_diagnostics import moving_block_means
from football_model.prematch_calibration import fit_calibrator,apply_calibrator

ROOT=REPOSITORY_ROOT
SOURCE=ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'
GROUPS=ROOT/'outputs/近期长期冲突诊断_20261026T034109989122Z'
CUTOFF=date(2025,8,1)


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def comparison(rows):
    if not rows: return {'n':0}
    p=np.asarray([r['original_probabilities'] for r in rows])
    q=np.asarray([r['research_calibrated_one_x_two'] for r in rows])
    y=np.asarray([r['actual_outcome'] for r in rows])
    truth=np.eye(3)[y]
    ll0=-np.log(p[np.arange(len(rows)),y]);ll1=-np.log(q[np.arange(len(rows)),y])
    b0=((p-truth)**2).sum(axis=1);b1=((q-truth)**2).sum(axis=1)
    deltas=np.column_stack((ll1-ll0,b1-b0))
    result={'n':len(rows),'original':probability_scores(p,y),'calibrated':probability_scores(q,y),
        'delta_log_loss':float((ll1-ll0).mean()),'delta_brier':float((b1-b0).mean()),
        'argmax_changed_n':int((p.argmax(axis=1)!=q.argmax(axis=1)).sum()),
        'class_bias_pp':{'original':((p-truth).mean(axis=0)*100).tolist(),
                         'calibrated':((q-truth).mean(axis=0)*100).tolist()},
        'calibration_bins_5':{'original':calibration_bins(p,y,bins=5),
                              'calibrated':calibration_bins(q,y,bins=5)},
        'conditional_4week_intervals':moving_block_means([r['date'] for r in rows],deltas,block_weeks=4)}
    valid=[r for r in rows if r.get('signal',{}).get('recent_supported_class') is not None]
    if valid:
        event=np.asarray([r['signal']['recent_supported_class'] for r in valid])
        actual=np.asarray([int(r['actual_outcome']==c) for r,c in zip(valid,event)])
        pv=np.asarray([r['original_probabilities'][c] for r,c in zip(valid,event)])
        qv=np.asarray([r['research_calibrated_one_x_two'][c] for r,c in zip(valid,event)])
        result['recent_supported_event']={'n':len(valid),'actual_frequency':float(actual.mean()),
            'original_mean':float(pv.mean()),'calibrated_mean':float(qv.mean()),
            'original_bias_pp':float((pv-actual).mean()*100),
            'calibrated_bias_pp':float((qv-actual).mean()*100),
            'mean_probability_change_pp':float((qv-pv).mean()*100)}
    return result


def run(lead,out):
    path=SOURCE/f'lead_{lead}/逐场完整预测.json'
    source=read(path)
    original=source['fresh_performance']
    by_id={r['match_id']:r for r in original}
    if len(by_id)!=len(original): raise ValueError('Duplicate saved forecast ID')
    for r in original:
        if not r['fit_before_date']<=r['forecast_before_date']<r['date']:
            raise ValueError('Base forecast chronology invalid')
        score=r['actual_score']
        expected=0 if score[0]>score[1] else 2 if score[0]<score[1] else 1
        if r['actual_outcome']!=expected: raise ValueError('Outcome axis mismatch')
    earlier=[r for r in original if r['period']=='2024-25']
    later=[r for r in original if r['period']=='2025-26']
    if any(r['date']>=CUTOFF.isoformat() for r in earlier) or any(r['forecast_before_date']<CUTOFF.isoformat() for r in later):
        raise ValueError('Calibration chronology invalid')
    params=fit_calibrator(earlier,CUTOFF,'temperature')
    write(out/f'提前{lead}日_温度参数.json',params)
    group_rows=read(GROUPS/f'提前{lead}日_逐场分组与误差.json')['records']
    grouping={r['match_id']:r['signal'] for r in group_rows}
    q=apply_calibrator([r['probabilities'] for r in later],params)
    saved=[]
    for r,values in zip(later,q,strict=True):
        if r['match_id'] not in grouping: raise ValueError('Missing existing diagnostic group')
        saved.append({k:r[k] for k in ('match_id','date','period','competition','home','away',
            'fit_before_date','forecast_before_date','actual_outcome','actual_score')} |
            {'original_probabilities':r['probabilities'],'research_calibrated_one_x_two':values.tolist(),
             'signal':grouping[r['match_id']],'calibrator_before_date':CUTOFF.isoformat()})
    overall=comparison(saved)
    if params['parameters']['beta']>0 and overall['argmax_changed_n']:
        raise ValueError('Positive temperature must preserve top class')
    conflict=comparison([r for r in saved if r['signal']['group']=='conflict'])
    results={'horizon_days':lead,'calibrator':params,'calibration_n':len(earlier),'test_n':len(later),
        'overall':overall,'conflict':conflict,
        'by_league':{c:comparison([r for r in saved if r['competition']==c]) for c in ('E0','D1','SP1','I1','F1')},
        'zero_beta_uniform_limit':params['parameters']['beta']==0,
        'source_original_forecasts_replaced':False,'base_model_fits':0,'calibrator_fits':1,
        'score_grid_redefined':False,'current_target_predictions_generated':False}
    write(out/f'提前{lead}日_较晚校准概率.json',{'records':saved})
    write(out/f'提前{lead}日_完整结果.json',results)
    if read(path)['fresh_performance']!=original: raise ValueError('Original forecast source changed')
    return results


def main():
    now=datetime.now(timezone.utc)
    out=ROOT/'outputs'/f'丰富模型单温度校准_{now.strftime("%Y%m%dT%H%M%S%fZ")}'
    out.mkdir(parents=True)
    plan={'created_at':now.isoformat(),'candidate':'existing fresh_performance121 ExtraTrees, saved time-forward forecasts',
        'prior_study_difference':'Existing temperature study used old regularized Poisson;this object has not been recalibrated.',
        'method':'Reuse prematch_calibration.fit_calibrator/apply_calibrator; q proportional to p^beta;beta=1/T>=0.',
        'earlier_calibration_period':'2024-25','fixed_before_date':CUTOFF.isoformat(),'later_period':'2025-26',
        'primary_horizon_days':3,'check_horizon_days':1,
        'parameter_scope':'one scalar per horizon shared across leagues and groups;no intercept,class bias or group coefficient',
        'primary_metric':'2025/26 full-cohort calibrated-minus-raw mean LogLoss',
        'secondary':['full-cohort Brier','unchanged Top1','class reliability bins',
                     'already defined conflict group event bias and whole-three-way scores','all five leagues'],
        'new_calibrator_fits':2,'new_base_fits':0,'parameter_grid_searches':0,
        'stopping':'Complete these two scalar fits and comparisons;do not add methods,thresholds,leaguespecific parameters or new horizons based on results.',
        'scope_warning':'Previously viewed historical seasons,not a new blind test. Conditional blocks do not include calibration parameter or base-training uncertainty. Horizon1/3 overlap.',
        'deployment':'No published prediction replaced. No calibrated score distribution implied. No current match outcomes or LLM probabilities used.',
        'sources':{'original_paper':'https://proceedings.mlr.press/v70/guo17a.html',
                   'official_documentation':'https://scikit-learn.org/stable/modules/calibration.html',
                   'author_demo_not_imported':'https://github.com/gpleiss/temperature_scaling'},
        'code_reuse':'Existing local NumPy/SciPy implementation;author PyTorch0.3 demo unmaintained,no new dependency or source code copy.'}
    write(out/'方法与停止点.json',plan)
    horizons={}
    for lead in (3,1):
        horizons[str(lead)]=run(lead,out)
        r=horizons[str(lead)]
        print(json.dumps({'lead':lead,'params':r['calibrator']['parameters'],'calibration_n':r['calibration_n'],
            'test_n':r['test_n'],'raw_ll':r['overall']['original']['log_loss'],
            'cal_ll':r['overall']['calibrated']['log_loss'],'dLL':r['overall']['delta_log_loss'],
            'dBrier':r['overall']['delta_brier'],'intervals':r['overall']['conditional_4week_intervals']['interval_95'],
            'conflict_bias':r['conflict']['recent_supported_event']},ensure_ascii=False),flush=True)
    write(out/'完整研究结果.json',{'method':plan,'horizons':horizons,'adopted':False})
    lines=['# 当前丰富模型：单参数概率校准','',
        '本轮实际对当前121项丰富模型的既有历史赛前概率做两次单参数拟合。2024/25学习，2025年8月1日固定，2025/26只应用；两个提前量分别学习，不合并重复比赛。0次基础模型重训，0轮参数网格，未改当前预测。',
        '', '以前做过的温度校准针对旧简单进球模型，不是此丰富模型。本次直接复用本地已实现的校准器；没有引入旧PyTorch示例或新依赖。',
        '', '| 提前量 | 较早学习场数 | 较晚检验场数 | 学到温度T | 原LL | 校准LL | LL差 | Brier差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for lead,r in horizons.items():
        o=r['overall'];T=r['calibrator']['parameters']['temperature']
        label=f'{T:.6f}' if T is not None else '均匀极限'
        lines.append(f"| {lead}日 | {r['calibration_n']} | {r['test_n']} | {label} | {o['original']['log_loss']:.6f} | {o['calibrated']['log_loss']:.6f} | {o['delta_log_loss']:+.6f} | {o['delta_brier']:+.6f} |")
    lines+=['','差值为校准减原样，负为更好。下面区间是已保存预测和已学参数条件下的4周时间分块95%描述区间，不是单场概率区间。',
        '', '| 提前量 | LL差区间 | Brier差区间 | 最高单项改变场数 |','|---|---|---|---:|']
    for lead,r in horizons.items():
        o=r['overall'];a,b=o['conditional_4week_intervals']['interval_95']
        lines.append(f"| {lead}日 | [{a[0]:+.6f}, {a[1]:+.6f}] | [{b[0]:+.6f}, {b[1]:+.6f}] | {o['argmax_changed_n']} |")
    lines+=['','## 已定义的近期／长期冲突组','',
        '| 提前量 | 场数 | 原近期支持单胜偏差 | 校准后偏差 | 该组LL差 |','|---|---:|---:|---:|---:|']
    for lead,r in horizons.items():
        c=r['conflict'];e=c['recent_supported_event']
        lines.append(f"| {lead}日 | {c['n']} | {e['original_bias_pp']:+.2f}个百分点 | {e['calibrated_bias_pp']:+.2f}个百分点 | {c['delta_log_loss']:+.6f} |")
    lines+=['','该组没有单独拟合参数；未根据较晚分组结果修改温度。单一全局温度不能专门解决某个分组或某一类别的偏差。',
        '', '## 五联赛完整保留','', '| 提前量／联赛 | 场数 | LL差 | Brier差 |','|---|---:|---:|---:|']
    for lead,r in horizons.items():
        for comp,v in r['by_league'].items():
            lines.append(f"| {lead}日／{comp} | {v['n']} | {v['delta_log_loss']:+.6f} | {v['delta_brier']:+.6f} |")
    lines+=['','## 使用边界与停止点','',
        '正温度只调概率的尖锐程度，保持主／平／客排序不变；即使命中率维持不变，概率评分仍可能改变。它不是提高三选一命中率的方法，也不加入神经网络。',
        '', '所有历史年份此前已被查看，仍属回溯研究。基础模型虽在较晚季继续正常前推更新，温度只由较早赛季的时间外预测学习；没有使用基础模型对其训练内比赛的预测拟合温度。',
        '', '只输出独立研究1X2向量，没有重新定义比分网格、总进球、让球或半全场分布，不能用这组1X2与旧比分概率混算。当前提前5—7天的模型适用性也不能由提前1／3日取代。',
        '', '本轮到两次校准拟合为止。无论结果如何，不根据后季反复尝试类别偏置、温度网格、联赛专用权重或提示词。当前原始预测保持不变，后续执行决定另列。',
        '', '方法参考：[原论文](https://proceedings.mlr.press/v70/guo17a.html)、[官方校准文档](https://scikit-learn.org/stable/modules/calibration.html)、[作者示例](https://github.com/gpleiss/temperature_scaling)。作者示例已不维护，本次仅参考方法并复用本地代码；外部研究不能替代本项目效果。']
    (out/'丰富模型校准结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'base_fits':0,'scalar_fits':2,'searches':0,'adopted':False},ensure_ascii=False))


if __name__=='__main__': main()

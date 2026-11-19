"""One earlier-fit conditional positive-draw joint-score research comparison."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import numpy as np

from conditional_positive_draw_20261026 import fit_adjustment, predict_adjusted, row_pieces
from diagnose_1x2_discrimination_20261025 import pair_rank_auc
from football_model.evaluation import calibration_bins
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from football_model.refresh_experiment import prediction_arrays, scores_for


ROOT=REPOSITORY_ROOT
SOURCE=ROOT/'outputs/当前121模型胜平负定位_20261026T051442339672Z'
CUTOFF='2025-08-01'
COMPS=('E0','D1','SP1','I1','F1')
CELLS=('0:0','1:1','other_positive_draw')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def scoring(rows):
    arrays=prediction_arrays(rows)
    losses={key:-np.log(p[np.arange(len(y)),y]) for key,(p,y) in arrays.items()}
    losses['exact_score']=np.asarray([r['exact_score_nll'] for r in rows])
    scores=scores_for(rows)
    scores['exact_score']={'n':len(rows),'log_loss':float(losses['exact_score'].mean())}
    return scores,losses


def class_summary(rows):
    p=np.asarray([r['probabilities'] for r in rows])
    y=np.asarray([r['actual_outcome'] for r in rows])
    top=p.argmax(axis=1)
    confusion=np.zeros((3,3),dtype=int)
    np.add.at(confusion,(y,top),1)
    result={'confusion_actual_rows_predicted_columns':confusion.tolist(),
            'calibration_bins_5':calibration_bins(p,y,bins=5),'by_class':{}}
    for c,name in enumerate(('home','draw','away')):
        actual=y==c
        result['by_class'][name]={'mean_probability':float(p[:,c].mean()),
            'actual_frequency':float(actual.mean()),'bias_pp':float((p[:,c]-actual).mean()*100),
            'auc':pair_rank_auc(actual,p[:,c]),'binary_brier':float(np.square(p[:,c]-actual).mean()),
            'top1_selected_n':int((top==c).sum()),'top1_correct_n':int(((top==c)&actual).sum())}
    return result


def comparison(old,new,*,intervals=False):
    if not old or [r['match_id'] for r in old]!=[r['match_id'] for r in new]:
        raise ValueError('Matched nonempty forecast cohorts required')
    os,ol=scoring(old); ns,nl=scoring(new)
    p=np.asarray([r['probabilities'] for r in old]);q=np.asarray([r['probabilities'] for r in new])
    y=np.asarray([r['actual_outcome'] for r in old]); truth=np.eye(3)[y]
    extra={
        'one_x_two_brier':(np.square(q-truth)-np.square(p-truth)).sum(axis=1),
        'top1_accuracy':(q.argmax(axis=1)==y).astype(float)-(p.argmax(axis=1)==y).astype(float),
        'draw_binary_brier':np.square(q[:,1]-(y==1))-np.square(p[:,1]-(y==1))}
    differences={k:nl[k]-ol[k] for k in ol}|extra
    columns=list(differences)
    matrix=np.column_stack(list(differences.values()))
    return {'n':len(old),'original':os,'adjusted':ns,
        'original_classes':class_summary(old),'adjusted_classes':class_summary(new),
        'difference_order':columns,
        'differences_adjusted_minus_original':dict(zip(columns,matrix.mean(axis=0).tolist())),
        'top1_changed_n':int((p.argmax(axis=1)!=q.argmax(axis=1)).sum()),
        'home_vs_away_direction_changed_n':int(((p[:,0]>p[:,2])!=(q[:,0]>q[:,2])).sum()),
        'four_week_conditional_intervals':moving_block_means([r['date'] for r in old],matrix,block_weeks=4)
            if intervals else None,'match_ids':[r['match_id'] for r in old]}


def score_cells(old,new):
    result={}
    for name in CELLS:
        actual=[]; p=[];q=[]
        for a,b in zip(old,new,strict=True):
            rates=a['base_rates']; p00=math.exp(-sum(rates));p11=p00*math.prod(rates)
            if name=='0:0':
                value=p00; event=a['actual_score']==[0,0]; factor=math.exp(-b['logZ'])
            elif name=='1:1':
                value=p11; event=a['actual_score']==[1,1];factor=math.exp(b['eta']-b['logZ'])
            else:
                value=a['probabilities'][1]-p00-p11
                event=a['actual_score'][0]==a['actual_score'][1] and a['actual_score'][0]>=2
                factor=math.exp(b['eta']-b['logZ'])
            p.append(value);q.append(value*factor);actual.append(event)
        result[name]={'actual_n':int(sum(actual)),'original_expected_n':float(sum(p)),
            'adjusted_expected_n':float(sum(q)),'original_bias_pp':float((np.asarray(p)-actual).mean()*100),
            'adjusted_bias_pp':float((np.asarray(q)-actual).mean()*100)}
    return result


def make_report(out,result):
    params=result['fit']; s=result['overall']; old=s['original'];new=s['adjusted'];d=s['differences_adjusted_minus_original']
    draw0=s['original_classes']['by_class']['draw'];draw1=s['adjusted_classes']['by_class']['draw']
    ci=dict(zip(s['difference_order'],s['four_week_conditional_intervals']['interval_95']))
    lines=['# 条件正比分平局：联合比分研究结果', '',
        '较早2024/25的1,726场只用于学习两个系数；2025年8月1日固定，后季2025/26的1,726场完整检验。'
        '保留当前121项模型与原参数日期，不重训球队、随机树或LLM。两季此前均已查看，仍属回溯研究。', '',
        '## 方法与复用', '',
        '在每场原联合比分上，对1:1、2:2等正比分平局乘以同场系数，再对整个比分分布统一归一化。'
        '该系数由截距和赛前强弱差共同决定，允许增加也允许减少；0:0不单独乘该系数。', '',
        '`d = abs(log(goal专家主队预期进球 / 客队预期进球))`', '',
        '`eta = beta0 + beta1*d; Q(h,a) = P0(h,a)*exp(eta*I[h=a>=1])/Z`', '',
        f"本次学得beta0={params['beta'][0]:.6f}，beta1={params['beta'][1]:.6f}。"
        '一次零起点BFGS，以完整比分概率损失学习；没有网格、重启、联赛专用参数或额外提前量。', '',
        '借鉴[footBayes的对角比分机制](https://raw.githubusercontent.com/LeoEgidi/footBayes/master/man/stan_foot.Rd)，'
        '本轮是本项目的两参数条件化改编，并非将其原R/Stan模型原样导入。'
        '直接复用了本地既有比分网格、概率汇总和SciPy优化接口，没有安装新依赖。', '',
        '## 后季完整结果', '',
        '| 指标 | 原模型 | 条件联合比分 |', '|---|---:|---:|',
        f"| 胜平负LogLoss | {old['one_x_two']['log_loss']:.6f} | {new['one_x_two']['log_loss']:.6f} |",
        f"| 胜平负Brier | {old['one_x_two']['brier_sum']:.6f} | {new['one_x_two']['brier_sum']:.6f} |",
        f"| 单选命中率 | {old['one_x_two']['accuracy']:.2%} | {new['one_x_two']['accuracy']:.2%} |",
        f"| 平局平均概率 | {draw0['mean_probability']:.2%} | {draw1['mean_probability']:.2%} |",
        f"| 实际平局比例 | {draw0['actual_frequency']:.2%} | {draw1['actual_frequency']:.2%} |",
        f"| 平局AUC | {draw0['auc']:.6f} | {draw1['auc']:.6f} |",
        f"| 平局二项Brier | {draw0['binary_brier']:.6f} | {draw1['binary_brier']:.6f} |",
        f"| 平局列首选场数 | {draw0['top1_selected_n']} | {draw1['top1_selected_n']} |",
        f"| 完整比分NLL | {old['exact_score']['log_loss']:.6f} | {new['exact_score']['log_loss']:.6f} |",
        f"| 总球0/1/2/3/4+ NLL | {old['total_goals_0_1_2_3_4plus']['log_loss']:.6f} | {new['total_goals_0_1_2_3_4plus']['log_loss']:.6f} |", '',
        'LogLoss、Brier及NLL越低越好；AUC是排序辨别能力，不是命中率。', '',
        f"胜平负LL差为{d['one_x_two']:+.6f}，4周条件区间[{ci['one_x_two'][0]:+.6f}, {ci['one_x_two'][1]:+.6f}]。"
        f"Brier差为{d['one_x_two_brier']:+.6f}，区间[{ci['one_x_two_brier'][0]:+.6f}, {ci['one_x_two_brier'][1]:+.6f}]。"
        '区间只反映既有预测及已拟合系数条件下的日历分块波动，未包含重新训练和系数估计的全部不确定性；AUC只列点值。', '',
        '## 既定分组：不按结果挑选', '',
        '| 范围 | 场数 | 胜平负LL差 | Brier差 | 平局AUC原→新 | 平局偏差原→新 |',
        '|---|---:|---:|---:|---:|---|']
    for name,g in result['groups'].items():
        a=g['original_classes']['by_class']['draw'];b=g['adjusted_classes']['by_class']['draw'];dg=g['differences_adjusted_minus_original']
        lines.append(f"| {name} | {g['n']} | {dg['one_x_two']:+.6f} | {dg['one_x_two_brier']:+.6f} | "
                     f"{a['auc']:.4f}→{b['auc']:.4f} | {a['bias_pp']:+.2f}→{b['bias_pp']:+.2f}pp |")
    for name,g in result['by_league'].items():
        a=g['original_classes']['by_class']['draw'];b=g['adjusted_classes']['by_class']['draw'];dg=g['differences_adjusted_minus_original']
        lines.append(f"| {name} | {g['n']} | {dg['one_x_two']:+.6f} | {dg['one_x_two_brier']:+.6f} | "
                     f"{a['auc']:.4f}→{b['auc']:.4f} | {a['bias_pp']:+.2f}→{b['bias_pp']:+.2f}pp |")
    lines += ['', '## 各类平局的数量表达', '',
        '| 比分 | 实际场数 | 原预期场数 | 新预期场数 |', '|---|---:|---:|---:|']
    for name,v in result['score_cells'].items():
        lines.append(f"| {name} | {v['actual_n']} | {v['original_expected_n']:.1f} | {v['adjusted_expected_n']:.1f} |")
    m=result['prediction_integrity']
    lines += ['', '## 能改变什么，不能改变什么', '',
        f"本轮最高选项改变{s['top1_changed_n']}场，主胜与客胜之间的方向改变{s['home_vs_away_direction_changed_n']}场。"
        '该结构保持主胜／客胜概率比，因此不能纠正胜负方向选反；1:1与2:2等正比分平局内部相对比例也保持不变。'
        '它主要改变平局与非平局之间的表达，没有新增球员、伤停或战术信息。', '',
        f"最终主客预期进球的平均绝对变化分别为{m['mean_absolute_expected_goal_change'][0]:.6f}、"
        f"{m['mean_absolute_expected_goal_change'][1]:.6f}。"
        '全部总球及让球概率由同一个新网格重算，不能把新均值再当作Poisson参数重新评分。'
        '完整比分NLL使用原解析Poisson NLL加logZ减实际正平局倾斜量，覆盖超出报告网格的高比分；'
        'S来自已保存概率，报告网格与此完整支持表达之间仅有原有可忽略尾截断差异。', '',
        '## 本轮决定与停止点', '',result['decision']['explanation'], '',
        '到这一组两参数学习和既定后季完整比较为止；不根据结果增加形状参数、改变窗口、拆联赛训练或再扫其他分布。'
        '当前比赛正式模型、LLM结果和已发布预测均未替换；没有复核当前赛果。']
    (out/'联合比分结果与决定.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    now=datetime.now(timezone.utc);tick=time.perf_counter()
    out=args.resume.resolve() if args.resume else ROOT/'outputs'/now.strftime('条件正平局联合比分研究_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=bool(args.resume))
    method={'created_at':now.isoformat(),'source':str(SOURCE),'candidate':'conditional_positive_draw_joint_score_tilt',
        'user_choice':'Study draw identification and joint scores; preserve original model',
        'fit_period':'2024-25','fixed_before_date':CUTOFF,'evaluate_period':'2025-26','horizon_days':3,
        'expected_n_each_period':1726,'parameters':['intercept','absolute_log_goal_expert_rate_ratio'],
        'law':'Q=P0*exp((b0+b1*gap)*I(h=a>=1))/Z; gap from earlier goal expert; no parameter sign restriction',
        'learning':'one equal-match full-score NLL BFGS fit, analytic gradient, zero initialization',
        'primary':'2025/26 all-target 1X2 LogLoss adjusted minus original',
        'secondary':['1X2 Brier/top1','draw AUC/calibration/Brier','exact-score and total-goal NLL',
                     'same-grid away-minus-two','all five leagues and existing ratio1.5 groups'],
        'limits':['Earlier layer fit period is in-sample for new coefficients, not another validation season',
                  'Already-viewed historical seasons, not fresh blind evaluation',
                  'New goal means change; H/A probability ratio and conditional positive-draw score ratios remain fixed',
                  'No new team/player information; performance does not establish causal error source',
                  'Conditional block intervals exclude full retraining and coefficient uncertainty',
                  'Date-level source availability, no verified original publication/revision timestamps',
                  'Current 5-7day target scope differs from historical3day scope'],
        'sources':{'concept':'https://raw.githubusercontent.com/LeoEgidi/footBayes/master/man/stan_foot.Rd',
                   'optimizer':'https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html'},
        'reuse':'Adapt diagonal-inflation concept; reuse local score_grid,FootballProbabilityGrid,summarize,SciPy; '
                'not footBayes unchanged model; no new dependency or copied external source',
        'stopping':'one two-parameter fit; zero team/tree/LLM refits; zero grids/restarts/new horizons; '
                   'finish later full cohort and all declared groups then stop without searching variants',
        'forecast_replacement':False,'current_results_checked':False}
    if not (out/'分析前方法.json').exists():write(out/'分析前方法.json',method)
    print(json.dumps({'output':str(out),'phase':'prepare'},ensure_ascii=False),flush=True)
    rows=read(SOURCE/'提前3日_已有预测与固定分组.json')['records']
    train=[r for r in rows if r['period']=='2024-25'];test=[r for r in rows if r['period']=='2025-26']
    if len(train)!=1726 or len(test)!=1726 or len({r['match_id'] for r in rows})!=len(rows):
        raise ValueError('Expected unique complete historical cohorts')
    train.sort(key=lambda r:(r['date'],r['match_id']));test.sort(key=lambda r:(r['date'],r['match_id']))
    param_path=out/'较早学习的两参数.json'
    if param_path.exists():
        params=read(param_path);new_fits=0
        if params['training_match_ids']!=[r['match_id'] for r in train] or params['before_date']!=CUTOFF:
            raise ValueError('Saved parameter research cohort differs')
    else:
        try:params=fit_adjustment(train,before_date=CUTOFF)
        except Exception as exc:
            write(out/'拟合失败.json',{'error':str(exc),'type':type(exc).__name__,'restarts':0})
            raise
        write(param_path,params);new_fits=1
    print(json.dumps({'phase':'fitted','beta':params['beta'],'optimizer':params['optimizer'],
                      'new_fits':new_fits},ensure_ascii=False),flush=True)
    original=[];adjusted=[];mean_changes=[];max_base_probability_error=0.;max_ratio_error=0.
    context_keys=('match_id','date','period','competition','home','away','fit_before_date',
                  'forecast_before_date','horizon_days','actual_score','actual_outcome','strength_group')
    for i,row in enumerate(test):
        pieces=row_pieces(row)
        forecast=summarize(score_grid(*row['rich_rates'],rho=0))
        old={k:row[k] for k in context_keys}
        old.update(forecast=forecast,probabilities=list(forecast['one_x_two'].values()),
                   base_rates=row['rich_rates'],exact_score_nll=pieces['base_exact_score_nll'])
        value=predict_adjusted(row,params)
        new={**{k:row[k] for k in context_keys},**value,'score_law':'conditional_positive_draw_joint_score_tilt'}
        oldp=np.asarray(old['probabilities']);newp=np.asarray(new['probabilities'])
        max_base_probability_error=max(max_base_probability_error,float(np.max(np.abs(oldp-row['rich_probabilities']))))
        max_ratio_error=max(max_ratio_error,abs(newp[0]/newp[2]-oldp[0]/oldp[2]))
        mean_changes.append(np.asarray(new['adjusted_expected_goals'])-row['rich_rates'])
        if i==0:
            changed_score=dict(row,actual_score=[19,19])
            probe=predict_adjusted(changed_score,params)
            if not np.allclose(newp,probe['probabilities'],rtol=0,atol=1e-14):
                raise ValueError('Prediction probabilities used target actual outcome')
            reloaded=predict_adjusted(row,read(param_path))
            if not np.allclose(newp,reloaded['probabilities'],rtol=0,atol=1e-14):
                raise ValueError('Parameter serialization changed forecast')
        original.append(old);adjusted.append(new)
        if (i+1)%500==0:print(json.dumps({'phase':'predicted','n':i+1},ensure_ascii=False),flush=True)
    if max_base_probability_error>1e-10 or max_ratio_error>1e-10:
        raise ValueError('Original forecast or H/A ratio changed unexpectedly')
    overall=comparison(original,adjusted,intervals=True)
    groups={g:comparison([r for r in original if r['strength_group']==g],
                         [r for r in adjusted if r['strength_group']==g],intervals=True)
            for g in ('balanced','clear_favourite')}
    leagues={g:comparison([r for r in original if r['competition']==g],
                          [r for r in adjusted if r['competition']==g]) for g in COMPS}
    auc0=overall['original_classes']['by_class']['draw']['auc'];auc1=overall['adjusted_classes']['by_class']['draw']['auc']
    delta=overall['differences_adjusted_minus_original']['one_x_two']
    explanation=('完整胜平负概率损失未改善，尚不支持采用。' if delta>=0 else
                 '完整胜平负平均概率损失下降，但当前仅有已查看后季的一次回溯对照；仍保留为独立候选，不替换正式预测。')
    if auc1<=auc0:explanation+='平局排序辨别未改善，不能把平均平局比例更接近实际解释为识别能力提高。'
    else:explanation+='平局排序AUC点值提高，但未为其估计完整训练不确定性，也不能据此宣称命中率上升。'
    balanced=groups['balanced']
    if (balanced['adjusted_classes']['by_class']['draw']['auc'] <=
            balanced['original_classes']['by_class']['draw']['auc']):
        explanation+='本轮重点均势组的平局排序未改善，说明主要收益仍是概率量级表达，尚未解决该组逐场辨别。'
    if overall['differences_adjusted_minus_original']['total_goals_0_1_2_3_4plus']>0:
        explanation+='总进球概率评分变差；正比分平局共用系数也使1:1与较高平局无法分别校正。'
    result={'method':read(out/'分析前方法.json'),'fit':params,'new_fits_this_execution':new_fits,
        'overall':overall,'groups':groups,'by_league':leagues,'score_cells':score_cells(original,adjusted),
        'prediction_integrity':{'max_original_probability_replay_error':max_base_probability_error,
            'max_home_away_probability_ratio_error':max_ratio_error,'target_label_change_affects_probabilities':False,
            'parameter_json_roundtrip_verified':True,
            'mean_absolute_expected_goal_change':np.abs(mean_changes).mean(axis=0).tolist(),
            'max_absolute_expected_goal_change':np.abs(mean_changes).max(axis=0).tolist()},
        'decision':{'adopted':False,'explanation':explanation,'stopped':True,'additional_fits_planned':0},
        'seconds':time.perf_counter()-tick}
    write(out/'逐场统一比分预测.json',{'original':original,'adjusted':adjusted})
    write(out/'完整研究结果.json',result)
    make_report(out,result)
    print(json.dumps({'output':str(out),'original_scores':overall['original'],'adjusted_scores':overall['adjusted'],
        'differences':overall['differences_adjusted_minus_original'],'draw_auc':[auc0,auc1],
        'original_draw':overall['original_classes']['by_class']['draw'],
        'new_draw':overall['adjusted_classes']['by_class']['draw'],'decision':result['decision'],
        'new_fits':new_fits,'seconds':result['seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

"""One convex weight study and genuinely new prematch pools; no result downloads."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np

from football_model.domain import Fixture
from football_model.learned_pool import LearnedLinearPool
from football_model.multiline import learned_arithmetic_pool, multi_record, score_multi
from football_model.probability import score_grid, summarize
from football_model.score_reconciliation import reconcile_score_probabilities


ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/模型LLM融合研究_20261028'
TODAY = ROOT/'outputs/今日意甲预测_20261028'
ASSISTED = ROOT/'data/serie_a_adjusted_context_20261028/模型辅助LLM回答.json'
BJ = timezone(timedelta(hours=8))
ZH = {'AC Milan':'AC米兰','Bologna':'博洛尼亚','Parma':'帕尔马','Udinese':'乌迪内斯',
      'Venezia':'威尼斯','Internazionale':'国际米兰','Genoa':'热那亚','Juventus':'尤文图斯',
      'Monza':'蒙扎','Napoli':'那不勒斯'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')


def when(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def adapt(row):
    return {'match_id':row['match_id'], 'kickoff':row['fixture']['kickoff'],
            'statistical_generated_at':row['statistical']['generated_at'],
            'llm_generated_at':row['llm_independent']['generated_at'],
            'statistical_probabilities':row['statistical_probabilities'],
            'llm_probabilities':row['llm_probabilities'],
            'result_observed_at':row['result_available_at'], 'actual_outcome':row['actual_outcome'],
            'statistical_family':row['statistical_model_family'],
            'llm_family':'declared_independent_subjective_llm_actual_model_id_unknown',
            'llm_track':'llm_independent'}


def metrics(probabilities, actuals):
    p=np.asarray(probabilities,dtype=float); y=np.asarray(actuals,dtype=int)
    if p.shape != (len(y),3) or not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError('Need positive finite complete replay probabilities without smoothing')
    if not np.allclose(p.sum(axis=1),1.,rtol=0,atol=1e-12):
        raise ValueError('Replay probabilities must sum to one')
    unique=(p==p.max(axis=1,keepdims=True)).sum(axis=1)==1
    return {'n':len(y),'correct':int(((p.argmax(axis=1)==y)&unique).sum()),
            'top1_ties':int((~unique).sum()),
            'log_loss':float(-np.log(p[np.arange(len(y)),y]).mean()),
            'brier':float(np.square(p-np.eye(3)[y]).sum(axis=1).mean())}


def run():
    # Re-reading a completed study does not create new fits or refresh forecasts.
    if (OUT/'研究与当前联合结果.json').exists():
        return read(OUT/'研究与当前联合结果.json')
    source=read(OUT/'真实前瞻融合样本.json')
    original=source['records']; rows=[adapt(r) for r in original]
    now=datetime.now(timezone.utc)
    settled=[r for r in original if r['result_available_at'] is not None and when(r['result_available_at'])<now]
    italy=[r for r in settled if r['competition']=='Serie A']
    if not italy:
        raise ValueError('Need the already archived later Serie A cohort; do not fetch new results')
    replay_cutoff=min(when(r['forecast_generated_at']) for r in italy)
    earlier=LearnedLinearPool().fit(rows,replay_cutoff)
    replay=[]
    for r in italy:
        p=earlier.predict(r['statistical_probabilities'],r['llm_probabilities'],
            match_id=r['match_id'],kickoff=r['fixture']['kickoff'],as_of=r['forecast_generated_at'])
        replay.append({'match_id':r['match_id'],'actual_outcome':r['actual_outcome'],
            'statistical':r['statistical_probabilities'],'llm':r['llm_probabilities'],'joint':p,
            'historical_pair_generated_at':r['forecast_generated_at']})
    replay_scores={name:metrics([r[name] for r in replay],[r['actual_outcome'] for r in replay])
                   for name in ('statistical','llm','joint')}
    learner=LearnedLinearPool().fit(rows,now)
    params=learner.parameters()
    joblib.dump(earlier,OUT/'较早10场权重模型.joblib')
    joblib.dump(learner,OUT/'当前13场权重模型.joblib')
    write(OUT/'较早前缀权重.json',earlier.parameters())
    write(OUT/'当前学得权重.json',params)
    write(OUT/'留后重建检验.json',{'reconstructed_at_utc':datetime.now(timezone.utc).isoformat(),
        'historical_fit_cutoff':replay_cutoff.isoformat(),'training_n':earlier.parameters()['n_matches'],
        'records':replay,'metrics':replay_scores,
        'was_actually_published_before_these_matches':False,
        'scope':'reconstructed chronological research using already seen results, EPL-to-Serie-A transfer, not a fresh blind result'})
    base=read(TODAY/'前瞻记录_goal_simple.json')
    distributions={r['fixture']['match_id']:r for r in read(TODAY/'统计预测与比分分布.json')['records']}
    generated=datetime.now(timezone.utc).isoformat()
    records,details=[],[]
    for r in base['records']:
        f=Fixture(**(r['fixture']|{'kickoff':when(r['fixture']['kickoff'])}))
        tracks={name:value for name,value in r['tracks'].items() if value is not None and name!='joint_pool'}
        pool,reason=learned_arithmetic_pool(tracks['statistical'],tracks['llm_independent'],learner,f,generated)
        if pool is None:
            raise ValueError('No identified learned pool: '+str(reason))
        tracks['joint_pool']=pool
        record=multi_record(f,generated,tracks=tracks,
            missing_reasons={'llm_model_informed':'The changed-input assisted LLM remains a separate research component'})
        record['source_original_forecast']=str(TODAY/'前瞻记录_goal_simple.json')
        record['source_weight_samples']=str(OUT/'真实前瞻融合样本.json')
        records.append(record)
        s=distributions[f.match_id]['goal_simple']
        grid=reconcile_score_probabilities(score_grid(*s['rates']),pool['probabilities'])
        details.append({'fixture':r['fixture'],'statistical':tracks['statistical']['probabilities'],
            'independent_llm':tracks['llm_independent']['probabilities'],'learned_joint':pool['probabilities'],
            'joint_forecast':summarize(grid),'joint_score_grid':grid.grid.tolist(),
            'score_shape_note':'Retains original goal-model P(score|H/D/A); means and totals recomputed, not newly learned from LLM'})
    finished=datetime.now(timezone.utc)
    if any(finished>=when(r['fixture']['kickoff']) for r in records):
        raise ValueError('Do not publish a current probability after kickoff')
    write(OUT/'今日五场学得联合前瞻.json',{'assembled_at':generated,'completed_at_utc':finished.isoformat(),
        'records':records,'results':{},'primary_original_forecasts_unchanged':True,
        'weight_reference_n':params['n_matches'],'new_research_forecast':True})
    write(OUT/'今日五场联合初始评分.json',score_multi(records,{}))
    result={'computed_at_utc':finished.isoformat(),'weight_parameters':params,
        'earlier_prefix_parameters':earlier.parameters(),'later_reconstructed_metrics':replay_scores,
        'current_records':details,'new_scalar_weight_fits':2,'parameter_grid_runs':0,
        'new_base_forecast_fits':0,'new_result_lookups':0,'unique_settled_training_matches':len(settled),
        'current_n':len(records),'current_settled_n':0,
        'changed_input_assisted_llm_weight':None,
        'adoption':'new research pool; empirical weight is not a proven optimal allocation',
        'limits':['Only 13 settled unique matches; mixed EPL/Serie-A source policies and small sample limit transfer.',
                  'Unknown exact LLM runtime model IDs and declared isolation remain disclosed.',
                  'Replay later three results already seen; training and replay scores are not unbiased improvement claims.',
                  'Do not transfer independent-LLM weights to a model-assisted LLM with changed inputs.',
                  'No free intercept, class or league weights, temperature, market input or extra model search.']}
    write(OUT/'研究与当前联合结果.json',result)
    return result


def record_assisted_line():
    """Archive the changed-input component separately, without borrowing old w."""
    destination=OUT/'今日五场对手校正辅助LLM前瞻.json'
    if destination.exists():
        return read(destination)
    assisted=read(ASSISTED)
    exposure=assisted['exposure']
    if (exposure.get('historical_statistical_expectations') is not True or
            any(exposure.get(key) is not False for key in (
                'current_target_statistical_probabilities','market','other_llm','target_results'))):
        raise ValueError('Changed-input response exposure differs from its recorded policy')
    answers={r['fixture_id']:r for r in assisted['records']}
    source=read(TODAY/'前瞻记录_goal_simple.json')
    if len(answers)!=5 or set(answers)!={r['fixture']['match_id'] for r in source['records']}:
        raise ValueError('Need exactly the same five fixtures for the assisted component')
    generated=datetime.now(timezone.utc).isoformat()
    records=[]
    for row in source['records']:
        answer=answers[row['fixture']['match_id']]
        if answer['fixture']!=row['fixture']:
            raise ValueError('Assisted response fixture differs from original record')
        fixture=Fixture(**(row['fixture']|{'kickoff':when(row['fixture']['kickoff'])}))
        tracks={k:v for k,v in row['tracks'].items() if v is not None and k!='joint_pool'}
        tracks['llm_model_informed']={
            'fixture_id':fixture.match_id,
            'id':f'adjusted_context_llm:{fixture.match_id}:{assisted["generated_at_utc"]}',
            'generated_at':assisted['generated_at_utc'],'probabilities':answer['probabilities'],
            'model':f'opponent-adjusted historical-model context; actual model_id={assisted["model_id"]}',
            'exposed_to_statistical':True,'exposed_to_market':False}
        record=multi_record(fixture,generated,tracks=tracks,
            missing_reasons={'joint_pool':'No settled pairs for this changed model-assisted LLM input policy; independent-LLM w not transferred'})
        record['llm_assistance_scope']='Historical model-derived xG expectations seen; current target model probabilities not seen'
        record['llm_response_source']=str(ASSISTED)
        record['llm_fact_packet']=str(ASSISTED.parent/'带对手校正的事实包.json')
        records.append(record)
    if any(datetime.now(timezone.utc)>=when(r['fixture']['kickoff']) for r in records):
        raise ValueError('Do not publish the assisted line after kickoff')
    document={'assembled_at':generated,'records':records,'results':{},'unique_fixtures':5,
        'original_forecasts_preserved':True,'learned_weight_for_new_assisted_component':None}
    write(destination,document)
    write(OUT/'今日五场辅助LLM初始评分.json',score_multi(records,{}))
    return document


def report(result):
    assisted=read(ASSISTED) if ASSISTED.exists() else None
    assisted_by={} if assisted is None else {r['fixture_id']:r for r in assisted['records']}
    weight=result['weight_parameters']['weight'];early=result['earlier_prefix_parameters']['weight']
    percent=lambda p:'／'.join(f'{100*v:.1f}' for v in p)
    lines=['# 对手校正接线与模型／LLM权重学习','',
        '用户已明确授权加强联合并用数据学习两条路线占比。本轮复用真实赛前档案，不下载新赛果，不生成历史LLM填充样本。所有新文件在项目目录。','',
        '81组版本配对按比赛去重后为28场，已结算13场、无归档结果15场。同场保留最早可用双线版本；意甲同刻版本保留原声明的进球主线。LLM原回答及事实包可核，但精确模型身份unknown、全运行上下文未独立认证。','',
        '## 只学习一个比例','',
        '`P联合 = w × P进球模型 + (1-w) × P独立LLM`，主／平／客共享同一w。以较早实际赛前记录的平均胜平负LogLoss最小化，解一维凸目标，保留0与1端点；不是强制50:50，也不按后季命中数选配比。','',
        f'较早英超10场拟合：统计{100*early:.2f}%／LLM{100*(1-early):.2f}%。目前全部13场拟合：统计{100*weight:.2f}%／LLM{100*(1-weight):.2f}%。这些是当前小样本经验最优点，不是已证实最优的未来占比，也没有可信的精细配比精度。','',
        '较早10场→较晚意甲3场是今天重建的时间留后研究；三场赛果此前已看过，且跨联赛及来源政策迁移，不能补称昨天实际发布的联合或全新盲测。','',
        '|较晚3场重建检验|正确/3|LogLoss↓|Brier↓|','|---|---:|---:|---:|']
    for name,m in result['later_reconstructed_metrics'].items():
        lines.append(f'|{name}|{m["correct"]}|{m["log_loss"]:.6f}|{m["brier"]:.6f}|')
    lines+=['','## 今天真正新增的五场联合','',
        '北京时间10月29日凌晨；沿用今晨已保存的主线和独立LLM，学得权重后另存新前瞻，不改旧版本。三概率依次为主胜／平局／客胜，单位%。','',
        '|北京时间|比赛|原进球模型|原独立LLM|学得权重联合|新对手校正辅助LLM|',
        '|---|---|---:|---:|---:|---:|']
    for r in result['current_records']:
        f=r['fixture'];a=assisted_by.get(f['match_id']);dt=when(f['kickoff']).astimezone(BJ)
        lines.append(f'|{dt:%m-%d %H:%M}|{ZH[f["home"]]}—{ZH[f["away"]]}|{percent(r["statistical"])}|{percent(r["independent_llm"])}|{percent(r["learned_joint"])}|'+(percent(a['probabilities']) if a else '尚未生成')+'|')
    lines+=['','联合比分分布复用原进球模型的条件比分形状，再按新的H/D/A区域缩放并重算总球与均值；不是LLM已经学会了比分、进球相关性或临场模拟。完整网格保存在JSON。','',
        '## 对手校正真正接入了什么','',
        '10队近5场共50条球队观察，涉及42场不同比赛、16个日期；每个日期只用更早资料固定拟合一次xG攻防模型。进攻残差=实际xG−当时信息所对应期望，防守残差=实际xGA−期望（正为更差），净差不再当作独立第三证据。原始表现与对手、主客场、期望和残差同时保留。','',
        '这些过去期望是今天重建的，没有冒充当时真实发布的预测；源资料修订时点未知。尚未校正红牌、点球、比分状态、战术与完整跨赛事负荷，旧赛季与新赛季仍通过衰减共享参数。','',
        '新辅助LLM仅读新增事实包，没见当前目标概率、赔率或旧LLM；因看到历史模型期望，明确标模型辅助，不能当独立确认。它与旧独立LLM的差异同时包含输入变化和一次生成波动，不能全归因于对手校正。','',
        '**本轮学得w只用于对应的原独立LLM。新辅助LLM没有已结算同输入政策的历史样本，其融合占比保持未估计，不偷用旧w。** 新辅助判断按真实生成时间另存，待后续自身赛前记录结算再学习对应权重。','',
        '## 停止位置与下一步','',
        '本轮完成两次一维权重拟合（较早前缀及当前全样本）、一次16日期的解释资料重建、一次新辅助LLM判断和五场真正新联合。至此停止，不因权重端点或某场方向不满意继续换目标、加分类权重或搜索参数。','',
        '后续先按整批新增已结算样本更新同一目标与同一来源政策的权重，再检验更晚比赛；若改变LLM输入或统计模型，另列版本。没有把本轮训练分数称为验证准确率，也不把线路数量当比赛样本量。','',
        '[去重样本说明](../../outputs/模型LLM融合研究_20261028/样本说明.md)；[当前权重](../../outputs/模型LLM融合研究_20261028/当前学得权重.json)；[当前联合前瞻](../../outputs/模型LLM融合研究_20261028/今日五场学得联合前瞻.json)；[对手校正资料](../../data/serie_a_adjusted_context_20261028/训练与来源说明.md)。','']
    (OUT/'本轮联合学习与结果.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('run','report'),default='report')
    args=parser.parse_args()
    result=run() if args.mode=='run' else read(OUT/'研究与当前联合结果.json')
    if ASSISTED.exists():
        record_assisted_line()
    report(result)
    print(json.dumps({'weights':result['weight_parameters']['weights'],
        'earlier_prefix_weights':result['earlier_prefix_parameters']['weights'],
        'later_reconstructed_metrics':result['later_reconstructed_metrics'],
        'current_n':result['current_n'],'current_settled_n':0},ensure_ascii=False))

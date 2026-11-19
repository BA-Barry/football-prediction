"""Independent read-only arithmetic/chronology audit; never fits a learner."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np


ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/模型LLM融合研究_20261028'
TODAY = ROOT/'outputs/今日意甲预测_20261028'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def time(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    assert parsed.tzinfo is not None
    return parsed.astimezone(timezone.utc)


errors=[]


def close(left,right):
    difference=float(np.max(np.abs(np.asarray(left,dtype=float)-np.asarray(right,dtype=float))))
    errors.append(difference)
    assert difference<2e-12,(left,right,difference)


def metric(p,y):
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int)
    assert p.shape==(len(y),3) and np.all(p>0)
    unique=(p==p.max(axis=1,keepdims=True)).sum(axis=1)==1
    return {'n':len(y),'correct':int(((p.argmax(axis=1)==y)&unique).sum()),
        'top1_ties':int((~unique).sum()),'log_loss':float(-np.log(p[np.arange(len(y)),y]).mean()),
        'brier':float(np.square(p-np.eye(3)[y]).sum(axis=1).mean())}


def obj(truth_stat,truth_lm,w):
    mass=w*truth_stat+(1-w)*truth_lm
    delta=truth_stat-truth_lm
    return float(-np.log(mass).mean()),float(-(delta/mass).mean()),float(np.square(delta/mass).mean())


def summarize_grid(matrix):
    p=np.asarray(matrix,dtype=float)
    assert p.ndim==2 and np.all(p>=0) and np.isfinite(p).all()
    close(p.sum(),1.)
    h,a=np.indices(p.shape);total=h+a;difference=a-h
    hda=[p[h>a].sum(),p[h==a].sum(),p[h<a].sum()]
    handicap=[p[difference>2].sum(),p[difference==2].sum(),p[difference<2].sum()]
    return {'one_x_two':dict(zip(('home','draw','away'),hda)),
        'expected_goals':{'home':float((h*p).sum()),'away':float((a*p).sum())},
        'total_goals':{**{str(k):float(p[total==k].sum()) for k in range(4)},'4+':float(p[total>=4].sum())},
        'over_2_5':float(p[total>=3].sum()),'both_score':float(p[(h>0)&(a>0)].sum()),
        'away_minus_two_jincai':dict(zip(('win','draw','lose'),handicap)),
        'away_minus_two_asian':dict(zip(('win','push','lose'),handicap)),
        'fair_odds_unbuffered':{'away_win':1/hda[2],
            'away_minus_two_jincai_win':1/handicap[0],
            'away_minus_two_asian':(1-handicap[1])/handicap[0]}}


def compare_numeric_tree(left,right):
    if isinstance(left,dict):
        assert set(left)==set(right),(set(left),set(right))
        for key in left:compare_numeric_tree(left[key],right[key])
    else:
        close(left,right)


def main():
    inventory=read(OUT/'真实前瞻融合样本.json')
    records=inventory['records'];byid={r['match_id']:r for r in records}
    assert len(byid)==len(records)==28
    checks=[]
    for name,filename,n in [('earlier','较早前缀权重.json',10),('current','当前学得权重.json',13)]:
        parameters=read(OUT/filename)
        cutoff=time(parameters['fit_before'])
        history=sorted([r for r in records if r['result_available_at'] is not None
                        and time(r['result_available_at'])<cutoff],
                       key=lambda r:(time(r['result_available_at']),time(r['fixture']['kickoff']),r['match_id']))
        ids=[r['match_id'] for r in history]
        assert ids==parameters['reference_match_ids'] and len(ids)==n
        assert parameters['n_matches']==n and parameters['weights']=={'statistical':1.,'llm':0.}
        stat=[];lm=[]
        for r in history:
            assert max(time(r['statistical']['generated_at']),time(r['llm_independent']['generated_at']))<time(r['fixture']['kickoff'])<time(r['result_available_at'])<cutoff
            assert r['llm_audit']['raw_probabilities_and_generation_match']
            assert r['llm_audit']['exposed_to_statistical'] is False
            assert r['llm_audit']['exposed_to_market'] is False
            assert r['market_used_for_learning'] is False
            stat.append(r['statistical_probabilities'][r['actual_outcome']])
            lm.append(r['llm_probabilities'][r['actual_outcome']])
        stat,lm=np.asarray(stat),np.asarray(lm)
        left,right=obj(stat,lm,0.),obj(stat,lm,1.)
        close(left[0],parameters['endpoint_log_loss']['llm'])
        close(right[0],parameters['endpoint_log_loss']['statistical'])
        close([left[1],right[1]],parameters['endpoint_gradients'])
        close(right[0],parameters['training_log_loss'])
        close(right[1],parameters['gradient_at_solution'])
        close(right[2],parameters['curvature_at_solution'])
        assert right[1]<0 and right[2]>0
        # Convexity and a strictly negative right derivative prove the endpoint
        # minimum without another fit, optimiser, grid, or model selection.
        model=joblib.load(OUT/('较早10场权重模型.joblib' if name=='earlier' else '当前13场权重模型.joblib'))
        assert model.parameters()==parameters
        assert set(model.reference_match_ids)==set(ids) and model.weight==1.
        checks.append({'name':name,'n':n,'fit_before':parameters['fit_before'],
            'reference_ids_match':True,'forecast_and_result_chronology_passed':True,
            'endpoint_log_loss':[left[0],right[0]],'endpoint_gradients':[left[1],right[1]],
            'endpoint_statistical_weight':1.,'global_empirical_endpoint_minimum_proved_by_convexity':True,
            'saved_model_parameters_match':True})
    earlier=read(OUT/'较早前缀权重.json');current=read(OUT/'当前学得权重.json')
    replay=read(OUT/'留后重建检验.json')
    replay_ids={r['match_id'] for r in replay['records']}
    assert len(replay_ids)==3 and not replay_ids&set(earlier['reference_match_ids'])
    assert replay['was_actually_published_before_these_matches'] is False
    for r in replay['records']:
        s=byid[r['match_id']]
        assert s['competition']=='Serie A'
        assert r['actual_outcome']==s['actual_outcome']
        assert time(s['result_available_at'])>=time(earlier['fit_before'])
        assert max(time(byid[k]['result_available_at']) for k in earlier['reference_match_ids'])<time(r['historical_pair_generated_at'])
        assert r['statistical']==s['statistical_probabilities']
        assert r['llm']==s['llm_probabilities']
        assert r['joint']==r['statistical']
    for route in ('statistical','llm','joint'):
        computed=metric([r[route] for r in replay['records']],[r['actual_outcome'] for r in replay['records']])
        for key,value in computed.items():close(value,replay['metrics'][route][key])
    original=read(TODAY/'前瞻记录_goal_simple.json')
    originals={r['fixture']['match_id']:r for r in original['records']}
    current_doc=read(OUT/'今日五场学得联合前瞻.json')
    study=read(OUT/'研究与当前联合结果.json')
    details={r['fixture']['match_id']:r for r in study['current_records']}
    old_distributions={r['fixture']['match_id']:r['goal_simple']
                       for r in read(TODAY/'统计预测与比分分布.json')['records']}
    assert len(current_doc['records'])==len(details)==len(originals)==5
    assert current_doc['results']=={} and study['current_settled_n']==0
    assert study['weight_parameters']==current and study['earlier_prefix_parameters']==earlier
    assert study['changed_input_assisted_llm_weight'] is None
    for r in current_doc['records']:
        fid=r['fixture']['match_id'];old=originals[fid];detail=details[fid]
        assert fid not in current['reference_match_ids'] and fid not in replay_ids
        assert r['fixture']==old['fixture']==detail['fixture']
        for route in ('statistical','llm_independent','market'):
            assert r['tracks'][route]==old['tracks'][route]
        assert r['tracks']['llm_model_informed'] is None
        pool=r['tracks']['joint_pool']
        assert pool['weights']=={'statistical':1.,'llm':0.}
        assert pool['probabilities']==old['tracks']['statistical']['probabilities']
        assert pool['weight_source']=='earlier_prospective_log_loss_fit'
        assert time(current['fit_before'])<=time(pool['generated_at'])<time(r['fixture']['kickoff'])
        assert pool['generated_at']==r['as_of']
        assert time(current_doc['completed_at_utc'])<time(r['fixture']['kickoff'])
        assert time(old['tracks']['statistical']['generated_at'])<time(pool['generated_at'])
        assert time(old['tracks']['llm_independent']['generated_at'])<time(pool['generated_at'])
        assert detail['statistical']==old['tracks']['statistical']['probabilities']
        assert detail['independent_llm']==old['tracks']['llm_independent']['probabilities']
        assert detail['learned_joint']==pool['probabilities']
        summary=summarize_grid(detail['joint_score_grid'])
        compare_numeric_tree(summary,detail['joint_forecast'])
        close(list(summary['one_x_two'].values()),pool['probabilities'])
        # At this learned endpoint the whole score grid, not only its H/D/A
        # margins, must remain the original goal model's distribution.
        close(detail['joint_score_grid'],old_distributions[fid]['score_grid']['probabilities'])
        assert pool['learned_parameters']==current
        for key,route in [('statistical','statistical'),('llm','llm_independent')]:
            component=pool['components'][key]
            assert all(component[k]==v for k,v in r['tracks'][route].items())
    assisted=read(OUT/'今日五场对手校正辅助LLM前瞻.json')
    answer=read(ROOT/'data/serie_a_adjusted_context_20261028/模型辅助LLM回答.json')
    assisted_by={r['fixture_id']:r for r in answer['records']}
    assert len(assisted['records'])==5 and assisted['results']=={}
    assert assisted['learned_weight_for_new_assisted_component'] is None
    assert answer['exposure']['historical_statistical_expectations'] is True
    assert answer['exposure']['current_target_statistical_probabilities'] is False
    assert answer['exposure']['market'] is False and answer['exposure']['other_llm'] is False
    for r in assisted['records']:
        fid=r['fixture']['match_id'];old=originals[fid]
        assert r['fixture']==old['fixture']
        for route in ('statistical','llm_independent','market'):
            assert r['tracks'][route]==old['tracks'][route]
        informed=r['tracks']['llm_model_informed']
        assert informed['exposed_to_statistical'] is True and informed['exposed_to_market'] is False
        assert informed['generated_at']==answer['generated_at_utc']
        assert informed['probabilities']==assisted_by[fid]['probabilities']
        assert r['tracks']['joint_pool'] is None and 'not transferred' in r['missing_reasons']['joint_pool']
        assert time(r['as_of'])<time(r['fixture']['kickoff'])
    result={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
        'new_model_fits':0,'new_parameter_searches':0,'new_result_lookups':0,
        'source_unique_n':len(records),'earlier_and_current_weight_checks':checks,
        'reconstructed_later_evaluation_n':len(replay_ids),'replay_targets_excluded_from_earlier_fit':True,
        'replay_does_not_claim_actual_prior_joint_publication':True,'replay_metrics_recomputed':True,
        'current_unique_fixtures':5,'current_results':0,
        'original_component_probabilities_ids_and_times_preserved':True,
        'new_pool_generation_time_follows_fitting_and_precedes_kickoff':True,
        'joint_score_grids_hda_totals_means_handicap_recomputed':True,
        'endpoint_whole_score_grid_matches_original_goal_distribution':True,
        'saved_weight_model_parameters_match_json':True,
        'max_arithmetic_difference':max(errors),
        'assisted_llm_separate_changed_input_policy':True,
        'assisted_track_is_informed_by_historical_models_not_current_target_probabilities':True,
        'independent_llm_weight_not_transferred_to_assisted_component':True,
        'limits':['The audit verifies archive chronology and arithmetic, not full historical runtime isolation or exact model identities.',
            'The empirical endpoint on 13 mixed-policy matches does not establish that LLM predictions have no future value.',
            'Training loss and the already seen three-fixture replay are not fresh evidence of current improvement.',
            'Five repeated component records across cohorts remain five matches, not ten or more independent observations.']}
    (OUT/'独立核对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'独立核对与所学.md').write_text('\n'.join([
        '# 模型／LLM权重：独立核对与所学','',
        '核对通过。只读取真实前瞻样本、保存参数和模型、原概率及联合产物，没有fit、优化器重跑、参数网格或赛果查询。','',
        '较早10场及当前13场的训练ID、原预测生成时间、结果首次可用时间与严格截止一致；后三场意甲没有进入较早10场训练。保存模型参数与JSON逐项相符。','',
        f'10场均匀LogLoss在LLM端为{checks[0]["endpoint_log_loss"][0]:.6f}、统计端为{checks[0]["endpoint_log_loss"][1]:.6f}，右端导数{checks[0]["endpoint_gradients"][1]:.6f}；13场对应为{checks[1]["endpoint_log_loss"][0]:.6f}、{checks[1]["endpoint_log_loss"][1]:.6f}，右端导数{checks[1]["endpoint_gradients"][1]:.6f}。','',
        '目标是线性概率池的负对数似然，二阶导数非负。右端导数仍为负，足以证明区间[0,1]内经验最小值在统计权重1、LLM权重0；无需再次调参或强制非零LLM占比。两个端点都是实际学习结果，不应把它概括成LLM永久没有价值。','',
        '后三场重建分数与原概率相符，并明确标注联合并未在当时真实发布。今日五场的统计、原独立LLM及市场ID／概率／生成时间完整保留；新的联合生成时间在权重拟合后、开球前。w=1所以本轮联合三概率与统计主线一致，不能包装成新增准确率。','',
        f'五个比分网格的H/D/A、总球、均值、双方进球及让两球均重新计算一致，最大算术差{max(errors):.3g}。','',
        '新对手校正LLM确实接触历史统计模型期望，故标模型辅助；它没有看到当前目标三概率、市场或旧LLM。新输入政策没有对应已结算双线样本，已单列前瞻、联合为空，也没有借用旧独立LLM的权重。不同cohort共用的五场仍只算五场。','',
        '结论只涉及本次固定线性池、现有小样本和两种LLM输入政策。统计与LLM可能共享历史信息，潜在新增语境信息仍需通过真实赛前积累判断；不因端点结果强凑50:50，也不据此否定LLM信息提取价值。','']),encoding='utf-8')
    print(json.dumps({'status':result['status'],'max_difference':max(errors),'checks':checks},ensure_ascii=False))


if __name__=='__main__':
    main()

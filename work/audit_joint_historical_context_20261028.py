"""Read-only independent cohort/chronology/replay checks for joint context.

No fitting, downloads, source modifications or current result lookups.
"""
from collections import Counter
import csv
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy.stats import skellam

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/历史情境共同接入_20261028'
BASE = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练'
DATA = ROOT/'data/historical_context_20261028'
BLOCKS = {'performance':121,'schedule':40,'availability':14,'participation':22,'coach':16}
COMPS = {'E0':'GB1', 'D1':'L1', 'SP1':'ES1', 'I1':'IT1', 'F1':'FR1'}


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path,'rt',encoding='utf-8-sig') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def by_id(rows):
    result={r['match_id']:r for r in rows}
    assert len(result)==len(rows), 'Duplicate match IDs'
    return result


def instant(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def scores(predictions):
    ps=np.asarray([r['probabilities'] for r in predictions],dtype=float)
    ys=np.asarray([r['actual_outcome'] for r in predictions],dtype=int)
    assert ps.shape==(1726,3) and np.isfinite(ps).all() and (ps>=0).all()
    assert np.max(abs(ps.sum(axis=1)-1)) < 1e-10
    actual=ps[np.arange(len(ys)),ys]
    assert (actual>0).all()
    correct=int((ps.argmax(axis=1)==ys).sum())
    return {'n':len(ys),'correct':correct,'accuracy':correct/len(ys),
        'log_loss':float(-np.log(actual).mean()),
        'brier_sum':float(((ps-np.eye(3)[ys])**2).sum(axis=1).mean())}


def main():
    base=by_id(read(BASE/'扩样合并特征.json.gz')['rows'])
    schedule=by_id(read(DATA/'schedule_sources/8753场赛程数值特征.json.gz')['rows'])
    availability=by_id(read(DATA/'availability_sources/8753场历史可用性上下文.json.gz')['rows'])
    archived_snapshots={r['archive_observation']['path']:r['archive_observation']
                        for r in availability.values() if r['archive_observation'] is not None}
    coaches=by_id(read(DATA/'coach_sources/coach_context_lead3.json.gz')['rows'])
    joined=by_id(read(OUT/'整合训练输入.json.gz')['rows'])
    assert len(base)==8753 and all(set(x)==set(base) for x in (schedule,availability,coaches,joined))
    source_games={}
    with gzip.open(DATA/'coach_sources/games.csv.gz','rt',encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            assert row['game_id'] not in source_games
            source_games[row['game_id']]=row
    corrections=read(DATA/'coach_sources/verified_date_corrections.json')
    for fix in corrections:
        r=source_games[fix['game_id']]
        assert (r['date'],r['competition_id'],r['home_club_id'],r['away_club_id'])==(
            fix['archive_date'],fix['competition_id'],fix['home_club_id'],fix['away_club_id'])
        r['date']=fix['canonical_date']
    participation_names=sorted({key for r in availability.values()
        for side in (r['existing_participation_change_features'] or []) for key in side})
    assert len(participation_names)==22
    none_count=Counter();checks=Counter()
    held_ids={mid for mid,r in base.items() if r['period']=='2025-26'}
    assert len(held_ids)==1726
    for mid,b in base.items():
        s,a,c,j=schedule[mid],availability[mid],coaches[mid],joined[mid]
        assert b['forecast_before_date']==s['forecast_cutoff']==a['forecast_before_date']==c['forecast_before_date']==j['forecast_before_date']
        assert b['fit_before_date']<=b['forecast_before_date']<b['date']
        assert (date.fromisoformat(b['date'])-date.fromisoformat(b['forecast_before_date'])).days==3
        assert all(b[k]==j[k] for k in ('date','period','competition','home','away','actual_score','actual_outcome','fit_before_date','horizon_days'))
        cutoff=b['forecast_before_date'];cutoff_time=instant(cutoff+'T00:00:00+00:00')
        if a['archive_observation'] is not None:
            proxy=a['archive_observation']
            assert instant(proxy['availability_proxy_at'])<cutoff_time
            assert instant(proxy['availability_proxy_at'])==max(instant(proxy['filename_observed_at']),instant(proxy['repository_recorded_at']))
            checks['status_snapshot_proxy_strictly_earlier']+=1
        p=a['existing_participation_change_features']
        if p is None:
            p=[{key:None for key in participation_names} for _ in range(2)]
        diagnostics=a['existing_participation_diagnostics']
        if diagnostics is not None:
            for side in ('home','away'):
                diag=diagnostics[side]
                if diag['latest_minute_reference_date'] is not None:
                    assert diag['latest_minute_reference_date']<cutoff
                    checks['past_participation_last_date_checked']+=1
                assert mid not in diag['exact_reference_match_ids']
                checks['participation_target_id_excluded']+=1
                for ref in diag['reference_status_sources']:
                    if ref['snapshot'] is not None:
                        snap=archived_snapshots[ref['snapshot']]
                        assert instant(snap['availability_proxy_at'])<cutoff_time
                        if ref['forecast_before_date'] is not None:
                            assert ref['forecast_before_date']<cutoff
                            assert instant(snap['availability_proxy_at'])<instant(ref['forecast_before_date']+'T00:00:00+00:00')
                        checks['past_reference_status_proxy_earlier']+=1
        for i,side in enumerate(('home','away')):
            rawcoach=c['coach_context'][i]
            assert rawcoach['team']==b[side]
            assert len(rawcoach['features'])==8
            assert c.get('target_source_game_id') not in rawcoach['source_game_ids']
            for gid in rawcoach['source_game_ids']:
                game=source_games[gid]
                assert game['date']<cutoff
                assert game['competition_id']==COMPS[b['competition']]
                assert rawcoach['club_id'] in (game['home_club_id'],game['away_club_id'])
                checks['raw_coach_source_references_strictly_earlier']+=1
            if rawcoach['max_source_date'] is not None:
                assert rawcoach['max_source_date']<cutoff
            if rawcoach['last_observed_coach_name'] is not None:
                assert any(game['date']==rawcoach['last_match_date'] and
                    ((game['home_club_id']==rawcoach['club_id'] and game['home_club_manager_name']==rawcoach['last_observed_coach_name']) or
                     (game['away_club_id']==rawcoach['club_id'] and game['away_club_manager_name']==rawcoach['last_observed_coach_name']))
                    for game in (source_games[gid] for gid in rawcoach['source_game_ids']))
                checks['last_coach_label_supported_by_prior_raw_source']+=1
            cf={'ctx_coach_own_'+k:v for k,v in rawcoach['features'].items()}
            cf.update({'ctx_coach_opponent_'+k:v for k,v in c['coach_context'][1-i]['features'].items()})
            blocks={'performance':b['fresh_performance_features'][i],
                'schedule':s['context_numeric_side_features'][i],
                'availability':a['availability_count_features'][i],
                'participation':p[i], 'coach':cf}
            expected={}
            for name,block in blocks.items():
                assert len(block)==BLOCKS[name]
                assert not expected.keys() & block.keys()
                expected.update(block)
                none_count[name]+=sum(v is None for v in block.values())
            assert expected==j['side_features'][i] and len(expected)==213
            assert all(not isinstance(v,str) for v in expected.values())
            assert not any('target' in key or 'manager_name' in key or 'coach_name' in key for key in expected)
            checks['complete_213_side_dicts_exactly_reconstructed']+=1
            checks['original_121_values_unchanged']+=121
            if b['period']=='2021-22':
                for name,value in blocks['schedule'].items():
                    if any(token in name for token in ('formal_count','nonleague_count','unresolved_count','last_log1p_gap')):
                        assert value is None
                checks['earlier_season_schedule_missing_preserved_sides']+=1
    # The model outputs must exist after the independent input work above.
    result=read(OUT/'共同接入训练结果.json')
    predicted=read(OUT/'共同情境_2025-26逐场预测.json')
    earlier=read(BASE/'扩样_20250801_2025-26预测.json')
    assert [r['match_id'] for r in predicted]==[r['match_id'] for r in earlier]
    assert set(r['match_id'] for r in predicted)==held_ids
    for left,right in zip(predicted,earlier,strict=True):
        assert all(left[k]==right[k] for k in ('date','period','competition','home','away','actual_score','actual_outcome'))
    independently_scored={'expanded121':scores(earlier),'context213':scores(predicted)}
    for label,key in [('expanded121','old_expanded121'),('context213','joint_context213')]:
        reported=result[key]['one_x_two']
        for measure in ('accuracy','log_loss','brier_sum'):
            assert abs(independently_scored[label][measure]-reported[measure])<1e-12
    learner=joblib.load(OUT/'共同情境_20250801.joblib')
    params=learner.parameters()
    ordered=sorted(joined.values(),key=lambda r:(r['date'],r['match_id']))
    training=[r for r in ordered if r['date']<'2025-08-01']
    held=[r for r in ordered if r['match_id'] in held_ids]
    assert params['before_date']=='2025-08-01' and params['reference_match_ids']==[r['match_id'] for r in training]
    assert len(training)==7027 and not set(params['reference_match_ids'])&held_ids
    assert params['configuration']==read(BASE/'扩样_20250801_训练说明.json')['configuration']
    names=params['feature_names'];assert len(names)==213
    matrix=np.asarray([[np.nan if side[key] is None else side[key] for key in names]
                       for r in training for side in r['side_features']],dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        medians=np.nanmedian(matrix,axis=0)
    medians=np.nan_to_num(medians,nan=0.)
    imputer_difference=float(np.max(abs(medians-learner.imputer.statistics_)))
    assert imputer_difference<1e-12
    assert learner.imputer.indicator_.features_.tolist()==np.flatnonzero(np.isnan(matrix).any(axis=0)).tolist()
    replay=learner.predict_many([r['side_features'] for r in held],[date.fromisoformat(r['date']) for r in held])
    rate_difference=max(abs(a-b) for mean,row in zip(replay,predicted,strict=True) for a,b in zip(mean,row['rates'],strict=True))
    assert rate_difference<1e-12
    hda_difference=0.
    for row in predicted:
        h,a=row['rates']
        p=[float(skellam.sf(0,h,a)),float(skellam.pmf(0,h,a)),float(skellam.cdf(-1,h,a))]
        hda_difference=max(hda_difference,max(abs(x-y) for x,y in zip(p,row['probabilities'],strict=True)))
    assert hda_difference<1e-10
    final=joblib.load(OUT/'共同情境_20260801_研究候选.joblib')
    final_parameters=final.parameters()
    assert final_parameters['before_date']=='2026-08-01'
    assert final_parameters['reference_match_ids']==[r['match_id'] for r in ordered]
    assert len(set(final_parameters['reference_match_ids'])&held_ids)==1726
    assert result['final_candidate_used_for_held_scoring'] is False
    examples=ordered[-5:]
    final_rates=final.predict_many([r['side_features'] for r in examples],[date(2026,8,1)]*len(examples))
    assert all(math.isfinite(x) and x>0 for pair in final_rates for x in pair)
    audit={'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_cohort_n':len(base),'joint_cohort_n':len(joined),'feature_blocks':BLOCKS,
        'raw_unknown_cells_preserved':dict(none_count),'checks':dict(checks),
        'training_20250801_n':len(training),'held_n':len(held),'held_training_id_intersection':0,
        'target_manager_fields_used_as_features':False,'target_lineup_or_minutes_used_as_features':False,
        'source_date_corrections_applied':corrections,
        'training_only_median_max_difference':imputer_difference,
        'training_only_missing_indicators_match':True,
        'saved_scoring_model_rate_replay_max_difference':rate_difference,
        'untruncated_skellam_vs_saved_hda_max_difference':hda_difference,
        'independently_recomputed_scores':independently_scored,
        'final_candidate_contains_held_past_matches':1726,
        'final_candidate_used_for_their_scoring':False,
        'new_fits':0,'new_network_requests':0,'new_result_queries':0,
        'issues_found':[],
        'limitations':['Previously examined historical development, not untouched test or current accuracy.',
            'Historical source dates and snapshot proxies do not establish all original publications or revisions.',
            'Unknowns are preserved in raw input; the existing learner imputes training-only medians with missing indicators.',
            'Higher dimensional trees can change split opportunities; aggregate score differences are not causal context effects.',
            'Coach observations come only from earlier same-league matches, not exact appointments or target coach confirmation.']}
    (OUT/'独立核对.json').write_text(json.dumps(audit,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    before,after=independently_scored['expanded121'],independently_scored['context213']
    md=f'''# 历史情境共同接入：独立只读核对

核对完成UTC：{audit['completed_at_utc']}。本轮核对未拟合、未联网、未查新赛果，没有改动训练输入或原始资料。

## 8753场与213项输入

- 同一8,753场目标全部保留，身份无重复；主／客两视图共17,506份字典。
- 独立从四个源包逐字段拼出121原表现＋40赛程＋14FPL状态计数＋22较早参与变化＋16较早教练状态，全部与保存训练输入一致。原121项{checks['original_121_values_unchanged']:,}个单元逐值保留。
- 原始None按块保持，数量：{json.dumps(dict(none_count),ensure_ascii=False)}。2021/22没有全赛事覆盖的3,572侧实际计数仍None；二元覆盖为0不是已知零负荷。
- 特征中没有目标教练名称、目标同场执教标签、目标首发或目标分钟；22项参与变化使用较早比赛。目标源game_id仅作联结元数据，没有成为数值特征。

## 时间与来源核对

- 从原Transfermarkt比赛CSV检查{checks['raw_coach_source_references_strictly_earlier']:,}条教练来源引用，均为对应俱乐部、同联赛且日期严格早于F=D−3；目标自身ID无交集。已知教练标签逐个有较早原字段支持。
- 官方日期修正仅应用game3886463（2022-12-30→12-29），原CSV不改；修正来源及依据保留在详细JSON。没有按比分选择匹配。
- {checks['status_snapshot_proxy_strictly_earlier']}场FPL快照代理时间均早于F午夜；代理为文件名记录与仓库提交时间较晚者。较早分钟引用及状态引用也早于F。它们仍是事后保存的来源版本代理，不保证Git推送／首次医疗公告或全部修订时间已还原。

## 训练、评分与重放

- 用于2025/26评分的树训练只含2025-08-01之前7,027场；1,726个目标与训练ID交集0。原121比较输出与新213输出的目标顺序、日期、主客、标签完全一致。
- 所有训练中位数独立重算最大差{imputer_difference:g}，缺失指示列一致；未用测试资料拟合预处理。
- 保存评分模型对全部1,726场重放进球率最大差{rate_difference:g}；无截尾Skellam概率与保存H/D/A最大差{hda_difference:.3g}。
- 独立评分：原扩样121首选{before['correct']}/1726，LL {before['log_loss']:.9f}、Brier {before['brier_sum']:.9f}；共同情境213首选{after['correct']}/1726，LL {after['log_loss']:.9f}、Brier {after['brier_sum']:.9f}。与报告一致。
- 2026-08-01最终候选合法包含这1,726场过去比赛，但没有用于它们的上述评分；最终候选完整8,753个训练ID与保存元数据一致。

未发现本轮拼接／日期／训练评分混用。以上是接入与实际执行核对，不证明情境的因果效果、独立信息量或当前比赛准确率。历史季已用于研发，不能称新盲测。停止于一次完整独立核对，无新增参数搜索或救分训练。
'''
    (OUT/'独立核对.md').write_text(md,encoding='utf-8')
    print(json.dumps({'checks':dict(checks),'scores':independently_scored,
        'rate_replay_difference':rate_difference,'hda_difference':hda_difference,'issues_found':[]},ensure_ascii=False))


if __name__=='__main__':
    main()

"""Independent date/input and saved-fit audit; no fitting or current scores."""
import argparse
from collections import Counter
from datetime import date, datetime
import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson, skellam

from run_npxg_comparison_20261025 import load_source
from run_shot_increment_20261025 import write


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def feature_audit(out, rows, record_map):
    coverage=read(out/'数据接入与覆盖.json')
    schedule_dir=Path(coverage['schedule_directory'])
    seasons=read(schedule_dir/'四季覆盖汇总.json')['seasons']
    known_blocked=set()
    events={}
    for season in seasons:
        summary=read(season['summary_file'])
        known_blocked.update(str(r['event_id']) for r in summary.get('known_specials',[]))
        source=read(Path(season['events_file']).with_name('相邻赛季补全_逐队历史观察.json'))
        for club in source['clubs'].values():
            team=f"football-data:{club['competition']}:{club['independent_label']}"
            destination=events.setdefault((club['competition'],team),{})
            for event in club['events']:
                identifier=str(event['event_id'])
                if identifier in destination:
                    assert destination[identifier]==event
                destination[identifier]=event
    official=read(schedule_dir/'官方部分比赛补充说明_20261025.json')
    known_blocked.update(str(r['event_id']) for r in official['matches'])
    league_slugs={'E0':'eng.1','D1':'ger.1','SP1':'esp.1','I1':'ita.1','F1':'fra.1'}
    counts=Counter()
    window_refs=0
    old_source_refs=0
    keys={k:set(rows[0][k][0]) for k in ('basic_features','performance_features','allcomp_features')}
    omitted=coverage['omitted_league_schedule_fields']
    assert len(omitted)==12
    for row in rows:
        cutoff=date.fromisoformat(row['fit_before_date'])
        target=date.fromisoformat(row['date'])
        assert cutoff<=target
        assert row['actual_score']==list(record_map[row['match_id']].full_time)
        for side in (0,1):
            basic=row['basic_features'][side]
            performance=row['performance_features'][side]
            integrated=row['allcomp_features'][side]
            assert set(basic)==keys['basic_features'] and set(performance)==keys['performance_features']
            assert set(integrated)==keys['allcomp_features']
            assert all(performance[k]==v for k,v in basic.items())
            assert all(integrated[k]==v for k,v in performance.items())
            assert not set(omitted)&set(integrated)
            assert not set(omitted)&set(performance)
            assert integrated['is_home']==float(side==0)
            assert integrated['forecast_log1p_lead_days']==math.log1p((target-cutoff).days)
            assert not any('goal_rates' in k or 'blend' in k or 'actual' in k for k in integrated)
        for label in ('home','away'):
            history=row['feature_history']['history'][label]
            for identifier in history['used_match_ids']:
                original=record_map[identifier]
                assert original.match_date<cutoff and original.goal_fit_exclusion is None
                old_source_refs+=1
            meta=row['allcomp_history']['history'][label]
            by_id=events[(row['competition'],meta['team'])]
            interval=meta['contiguous_member_scope']
            earlier=[]
            if interval:
                begin=date.fromisoformat(interval[0])
                for identifier,event in by_id.items():
                    when=datetime.fromisoformat(event['date_utc_text'].replace('Z','+00:00')).date()
                    if begin<=when<cutoff and event['category']=='competitive':
                        earlier.append((when,identifier,event))
            good=sorted((r for r in earlier if r[2]['completed'] and r[1] not in known_blocked),key=lambda r:(r[0],r[1]))
            bad=[r for r in earlier if not r[2]['completed'] or r[1] in known_blocked]
            # Use scoring-side orientation independently of the feature builder.
            side_index=0 if label=='home' else 1
            prefix='own_allcomp_'
            features=row['allcomp_features'][side_index]
            for width,wm in meta['windows'].items():
                days=int(width)
                left=date.fromordinal(cutoff.toordinal()-days)
                chosen=[r for r in good if r[0]>=left]
                uncertain=[r for r in bad if r[0]>=left]
                covered=bool(interval) and date.fromisoformat(interval[0])<=left
                usable=covered and not uncertain
                assert wm['covered']==covered and wm['usable']==bool(usable)
                assert wm['selected_event_ids']==[r[1] for r in chosen]
                assert wm['selected_event_dates']==[r[0].isoformat() for r in chosen]
                assert [r['event_id'] for r in wm['excluded_or_unresolved_events']]==[r[1] for r in uncertain]
                assert features[prefix+f'window_usable_{days}d']==float(bool(usable))
                assert features[prefix+f'formal_count_{days}d']==(float(len(chosen)) if usable else None)
                nonleague=sum(r[2]['competition_slug']!=league_slugs[row['competition']]
                              or not r[2]['is_regular_league_fixture'] for r in chosen)
                assert features[prefix+f'nonleague_count_{days}d']==(float(nonleague) if usable else None)
                window_refs+=len(chosen)
                counts['unusable_windows']+=not usable
            last=good[-1] if good else None
            last_ok=last is not None and not any(r[0]>=last[0] for r in bad)
            expected=math.log1p((cutoff-last[0]).days) if last_ok else None
            assert features[prefix+'last_log1p_gap_to_cutoff_days']==expected
            recent=good[-3:]
            recent_ok=len(recent)==3 and not any(r[0]>=recent[0][0] for r in bad)
            expected=math.log1p(sum((b[0]-a[0]).days for a,b in zip(recent,recent[1:]))/2) if recent_ok else None
            assert features[prefix+'previous3_log1p_mean_interval_days']==expected
            counts['unknown_last']+=not last_ok
    return {'status':'passed','rows':len(rows),'feature_columns':{k:len(v) for k,v in keys.items()},
            'omitted_raw_league_gap_fields':omitted,'historical_performance_reference_entries':old_source_refs,
            'allcomp_window_reference_entries':window_refs,'missing_measurements':dict(counts),
            'comparison':'two rich schemes share every performance column and forecast lead; integrated adds only40schedule columns',
            'future_outcome_or_blend_inputs_found':False}


def independent_scores(track):
    rates=np.array([r['rates'] for r in track])
    actual=np.array([r['actual_score'] for r in track])
    y=np.where(actual[:,0]>actual[:,1],0,np.where(actual[:,0]<actual[:,1],2,1))
    h,a=rates[:,0],rates[:,1]
    p=np.column_stack((skellam.sf(0,h,a),skellam.pmf(0,h,a),skellam.cdf(-1,h,a)))
    total=np.column_stack([poisson.pmf(k,h+a) for k in range(4)]+[poisson.sf(3,h+a)])
    handicap=np.column_stack((skellam.cdf(-3,h,a),skellam.pmf(-2,h,a),skellam.sf(-2,h,a)))
    total_y=np.minimum(actual.sum(axis=1),4)
    hm=actual[:,1]-actual[:,0]-2
    handicap_y=np.where(hm>0,0,np.where(hm==0,1,2))
    exact=np.sum(rates-actual*np.log(rates)+gammaln(actual+1),axis=1)
    result={'exact_score':{'n':len(track),'log_loss':float(exact.mean())}}
    max_probability_error=0.
    for domain,probabilities,labels,saved in (
            ('one_x_two',p,y,np.array([r['probabilities'] for r in track])),
            ('total_goals_0_1_2_3_4plus',total,total_y,np.array([[r['forecast']['total_goals'][str(k)]
                 for k in range(4)]+[r['forecast']['total_goals']['4+']] for r in track])),
            ('away_minus_two_three_way',handicap,handicap_y,np.array([[r['forecast']['away_minus_two_jincai'][k]
                 for k in ('win','draw','lose')] for r in track]))):
        error=float(np.max(np.abs(probabilities-saved)))
        max_probability_error=max(max_probability_error,error)
        assert error<1e-10
        truth=np.eye(probabilities.shape[1])[labels]
        result[domain]={'n':len(track),'log_loss':float(-np.log(probabilities[np.arange(len(track)),labels]).mean()),
            'brier_sum':float(np.square(probabilities-truth).sum(axis=1).mean()),
            'rps':float(np.square(np.cumsum(probabilities,axis=1)[:,:-1]-np.cumsum(truth,axis=1)[:,:-1]).sum(axis=1).mean()/(probabilities.shape[1]-1)),
            # Analytic versus finite-grid ties may break by <1e-15; require
            # agreement away from ties, then replay the saved tie convention.
            'accuracy':float((saved.argmax(axis=1)==labels).mean())}
        ordered=np.sort(probabilities,axis=1)
        decisive=ordered[:,-1]-ordered[:,-2]>1e-10
        assert np.array_equal(probabilities.argmax(axis=1)[decisive],saved.argmax(axis=1)[decisive])
    return result,max_probability_error


def model_audit(out,rows,record_map):
    result=read(out/'完整方案结果汇总.json')
    parameters=read(out/'全部模型参数.json')
    predictions=read(out/'逐场完整预测.json')
    selection=read(out/'较早方案选择.json')
    arms=result['plan']['arms']
    expected_counts={'2024-25':1733,'2025-26':1732}
    assert len(parameters)==result['new_scheme_fits']==176
    expected_ids={period:[r['match_id'] for r in rows if r['period']==period] for period in expected_counts}
    row_map={r['match_id']:r for r in rows}
    refs=0
    for key,parameter in parameters.items():
        arm,cutoff=key.split(':',1)
        history=[r['match_id'] for r in rows if r['date']<cutoff]
        assert parameter['reference_match_ids']==history
        assert parameter['n_matches']==len(history)
        assert parameter['n_scoring_sides']==2*len(history)
        assert all(not record_map[k].goal_fit_exclusion for k in history)
        assert parameter['before_date']==cutoff and parameter['training_max_date']<cutoff
        expected_features=arms[arm][1]
        assert parameter['feature_names']==sorted(row_map[history[0]][expected_features+'_features'][0])
        assert parameter['pipeline_feature_set']==expected_features
        refs+=len(history)
    max_score_error,max_probability_error,score_checks=0.,0.,0
    for period,summary in (('2024-25',result['validation']),('2025-26',result['test'])):
        assert summary['n']==expected_counts[period]
        assert summary['match_ids']==expected_ids[period]
        for arm,saved_scores in summary['scores'].items():
            track=[r for r in predictions[arm] if r['period']==period]
            assert [r['match_id'] for r in track]==expected_ids[period]
            recomputed,p_error=independent_scores(track)
            max_probability_error=max(max_probability_error,p_error)
            for domain,values in recomputed.items():
                for statistic,value in values.items():
                    error=abs(value-saved_scores[domain][statistic])
                    max_score_error=max(max_score_error,error)
                    assert error<1e-10
            score_checks+=1
    # The calibration-year apparent-fit blend must not enter validation.
    assert 'clean_blend' not in result['validation']['scores']
    assert len(predictions['clean_blend'])==expected_counts['2025-26']
    assert selection['selected']==result['selected']==min(arms,key=lambda a:(result['validation']['scores'][a]['exact_score']['log_loss'],a))
    first_late=min((out/'models').glob('*_2025-08-01.joblib'),key=lambda p:p.stat().st_mtime)
    assert (out/'较早方案选择.json').stat().st_mtime<=first_late.stat().st_mtime
    max_replay_error=0.
    replayed=0
    for arm,(_,feature_set) in arms.items():
        later=[r for r in predictions[arm] if r['period']=='2025-26']
        cutoffs=sorted({r['fit_before_date'] for r in later})
        for cutoff in (cutoffs[0],cutoffs[-1]):
            selected=[r for r in later if r['fit_before_date']==cutoff][:3]
            model=joblib.load(out/'models'/f'{arm}_{cutoff}.joblib')
            rates=model.predict_many([row_map[r['match_id']][feature_set+'_features'] for r in selected],
                                     [date.fromisoformat(r['date']) for r in selected])
            for predicted,saved in zip(rates,selected,strict=True):
                error=max(abs(a-b) for a,b in zip(predicted,saved['rates'],strict=True))
                max_replay_error=max(max_replay_error,error)
                assert error<1e-10
                replayed+=1
    return {'status':'passed','saved_models':len(parameters),'checked_training_reference_entries':refs,
        'score_tracks_recomputed':score_checks,'max_analytic_probability_error':max_probability_error,
        'max_score_statistic_error':max_score_error,'replayed_fixture_rate_pairs':replayed,
        'max_saved_model_replay_error':max_replay_error,'earlier_selected':selection['selected'],
        'earlier_selection_precedes_late_fit':True,'validation_has_apparent_blend':False,
        'all_registered_incomplete_fixtures_excluded':True,
        'scope':'date-level retrospective computation; no claim complete publication-time reconstruction or unknown source perfection'}


def previous_failure_replay(out,rows):
    old_path=Path('G:/Codex/football-prediction-2026-10-22/outputs/多信息多算法前推_20261025T060704977450Z/丰富线性外推失误诊断.json')
    old=read(old_path)
    predictions=read(out/'逐场完整预测.json')
    result=read(out/'完整方案结果汇总.json')
    row_map={r['match_id']:r for r in rows}
    by_arm={a:{r['match_id']:r for r in values} for a,values in predictions.items()}
    cases=[]
    for case in old['cases']:
        identifier=case['match_id']
        row=row_map[identifier]
        cases.append({'match_id':identifier,'date':row['date'],'actual_score':row['actual_score'],
            'old_131field_glm_rates':[s['rate'] for s in case['sides']],
            'old_observed_sameleague_intervals_days':[s['previous_same_league_interval_days'] for s in case['sides']],
            'new_scheme_rates':{arm:by_arm[arm][identifier]['rates'] for arm in result['plan']['arms']},
            'new_schedule_last_events':{side:row['allcomp_history']['history'][side]['last_completed_event']
                for side in ('home','away')}})
    maxima={}
    for arm in result['plan']['arms']:
        later=[r for r in predictions[arm] if r['period']=='2025-26']
        values=[(rate,r,i) for r in later for i,rate in enumerate(r['rates'])]
        maximum=max(values,key=lambda x:x[0])
        quantiles=np.quantile([v[0] for v in values],[.5,.95,.99]).tolist()
        maxima[arm]={'n_scoring_sides':len(values),'maximum_rate':maximum[0],
            'match_id':maximum[1]['match_id'],'date':maximum[1]['date'],'side':'home' if maximum[2]==0 else 'away',
            'actual_score':maximum[1]['actual_score'],'quantiles_50_95_99':quantiles,
            'count_above5':sum(v[0]>5 for v in values),'count_above10':sum(v[0]>10 for v in values)}
    return {'scope':'saved historical rate comparison only; no fitting/clipping/current result queries',
        'old_diagnostic':str(old_path),'cases':cases,'all_later_scheme_extrema':maxima,
        'interpretation':'observes rates after source correction, removal of ambiguous raw league-gap fields and new scheme rebuild; not causal attribution to one change and not used to select a later winner'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--features-only',action='store_true')
    args=parser.parse_args()
    out=args.directory
    rows=read(out/'完整历史特征.json')['rows']
    _,independent,*_=load_source()
    features=feature_audit(out,rows,independent)
    write(out/'独立完整输入核对.json',features)
    print(json.dumps(features,ensure_ascii=False),flush=True)
    if args.features_only:
        return
    models=model_audit(out,rows,independent)
    write(out/'独立模型与全概率评分核对.json',models)
    print(json.dumps(models,ensure_ascii=False),flush=True)
    failures=previous_failure_replay(out,rows)
    write(out/'旧极端样本与新方案均值检查.json',failures)
    print(json.dumps({'new_mean_maxima':{a:r['maximum_rate'] for a,r in failures['all_later_scheme_extrema'].items()},
                      'compared_old_cases':len(failures['cases'])},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

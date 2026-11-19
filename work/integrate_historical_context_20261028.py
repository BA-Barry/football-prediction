"""One joint context integration with the existing goal learner; no grid.

This is examined historical development data. Two fits only: a fixed earlier
cutoff evaluation and a final research candidate. Live forecasts are untouched.
"""
from collections import Counter
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import time

import joblib

from football_model.early_goal_learners import EarlyGoalLearner
from run_broad_early_models_20261025 import predict_row
from run_shot_increment_20261025 import score_details

ROOT = REPOSITORY_ROOT
BASE = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练'
DATA = ROOT/'data/historical_context_20261028'
OUT = ROOT/'outputs/历史情境共同接入_20261028'
FILES = {'base': BASE/'扩样合并特征.json.gz',
    'schedule': DATA/'schedule_sources/8753场赛程数值特征.json.gz',
    'availability': DATA/'availability_sources/8753场历史可用性上下文.json.gz',
    'coach': DATA/'coach_sources/coach_context_lead3.json.gz'}
HELD_CUTOFF = date(2025, 8, 1)
FINAL_CUTOFF = date(2026, 8, 1)
CONTEXT = ('match_id', 'date', 'period', 'competition', 'home', 'away',
           'fit_before_date', 'forecast_before_date', 'horizon_days', 'actual_score', 'actual_outcome')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def index(rows):
    result = {r['match_id']:r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate fixture identity in context rows')
    return result


def numeric(values):
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                             or not math.isfinite(v)) for v in values.values()):
        raise ValueError('Context features require finite numbers or explicit unknowns')
    return values


def merge_side(original, additions):
    result = dict(original)
    for block in additions:
        if result.keys() & block.keys():
            raise ValueError('A context field would overwrite an existing measurement')
        result.update(numeric(block))
    return result


def assemble(originals, schedules, availability, coaches):
    all_ids = set(index(originals))
    if any(set(x) != all_ids for x in (schedules, availability, coaches)):
        raise ValueError('All 8753 fixtures must remain, including context unknowns')
    participation_names = set()
    for row in availability.values():
        for side in row.get('existing_participation_change_features') or []:
            participation_names.update(side)
    if len(participation_names) != 22:
        raise ValueError('Existing participation block differs from its documented 22 fields')
    rows, coverage = [], Counter()
    for base in originals:
        ident, cutoff = base['match_id'], base['forecast_before_date']
        s, a, c = schedules[ident], availability[ident], coaches[ident]
        if s['forecast_cutoff'] != cutoff or any(x['forecast_before_date'] != cutoff for x in (a,c)):
            raise ValueError('Context and performance must use the same forecast cutoff')
        for item in (a,c):
            if any(item[k] != base[k] for k in ('date','home','away','competition','period')):
                raise ValueError('Cross-provider context identity differs')
        if not base['fit_before_date'] <= cutoff < base['date']:
            raise ValueError('Earlier-date feature rule changed')
        observation = a['archive_observation']
        if observation is not None:
            proxy = datetime.fromisoformat(observation['availability_proxy_at'].replace('Z','+00:00'))
            before = datetime.fromisoformat(cutoff+'T00:00:00+00:00')
            if proxy >= before:
                raise ValueError('Status snapshot arrived after the forecast cutoff')
        for side in c['coach_context']:
            if side['max_source_date'] is not None and side['max_source_date'] >= cutoff:
                raise ValueError('Coach context used a match at or after forecast cutoff')
        p = a.get('existing_participation_change_features')
        if p is None:
            p = [{key:None for key in participation_names} for _ in range(2)]
        features = []
        for side in range(2):
            coach_features = {'ctx_coach_own_'+k:v for k,v in c['coach_context'][side]['features'].items()}
            coach_features.update({'ctx_coach_opponent_'+k:v for k,v in c['coach_context'][1-side]['features'].items()})
            features.append(merge_side(base['fresh_performance_features'][side],
                [s['context_numeric_side_features'][side], a['availability_count_features'][side],
                 p[side], coach_features]))
        if any(len(side) != 213 for side in features):
            raise ValueError('Expected 121 performance +40 schedule +14 status +22 participation +16 coach fields')
        if any(any(side[k] != features[i][k] for k in side)
               for i,side in enumerate(base['fresh_performance_features'])):
            raise ValueError('An original performance feature changed')
        coverage['all_matches'] += 1
        coverage['both_availability_teams_observed'] += all(t['observed'] for t in a['teams'])
        coverage['both_latest_coach_labels_known'] += all(t['features']['latest_coach_label_known'] == 1 for t in c['coach_context'])
        rows.append({k:base[k] for k in CONTEXT} | {'side_features':features,
            'context_source_match_id':ident})
    rows.sort(key=lambda r:(r['date'],r['match_id']))
    return rows, dict(coverage)


def prepare():
    base = read(FILES['base'])['rows']
    if len(base) != 8753:
        raise ValueError('Expanded historical cohort changed')
    return assemble(base, index(read(FILES['schedule'])['rows']),
        index(read(FILES['availability'])['rows']), index(read(FILES['coach'])['rows']))


def fit(rows, cutoff, name):
    path = OUT/f'{name}.joblib'
    before = [r for r in rows if r['date'] < cutoff.isoformat()]
    tick = time.perf_counter()
    if path.exists():
        learner, action = joblib.load(path), 'reused'
    else:
        learner = EarlyGoalLearner('extra_trees','rich').fit(before, cutoff)
        joblib.dump(learner,path,compress=3)
        action = 'fitted'
    parameters = learner.parameters()
    original = read(BASE/'扩样_20250801_训练说明.json')
    if (parameters['reference_match_ids'] != [r['match_id'] for r in before]
            or parameters['before_date'] != cutoff.isoformat()
            or parameters['configuration'] != original['configuration']
            or parameters['half_life_days'] != 180 or len(parameters['feature_names']) != 213):
        raise ValueError('The reused learner architecture, earlier cohort or parameters differ')
    elapsed = time.perf_counter()-tick
    write(OUT/f'{name}_训练说明.json',parameters | {'source_files':{k:str(v) for k,v in FILES.items()},
        'action_this_execution':action,'elapsed_seconds':elapsed,
        'historical_first_publication_reconstructed':False})
    print(json.dumps({'fit':name,'action':action,'n':len(before),'seconds':elapsed},ensure_ascii=False),flush=True)
    return learner, action


def predictions(model, rows):
    rates = model.predict_many([r['side_features'] for r in rows],
                               [date.fromisoformat(r['date']) for r in rows])
    return [predict_row({k:r[k] for k in ('match_id','date','period','competition','home','away',
        'actual_score','actual_outcome')}, mean) for r,mean in zip(rows,rates,strict=True)]


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    plan_path = OUT/'共同接入方法.json'
    if not plan_path.exists():
        write(plan_path, {'created_at':datetime.now(timezone.utc).isoformat(),
            'purpose':'One joint integration of context, existing performance retained; not independent single-factor trials',
            'algorithm':'Existing 120-tree ExtraTrees full-goal model; fixed depth6 leaf60 features0.8 seed20261025,180day decay',
            'features':{'performance':121,'past_schedule':40,'archived_status':14,
                'past_participation_state_change':22,'lagged_coach_context':16,'total':213},
            'new_fits_planned':2,'parameter_searches':0,'feature_selection_searches':0,
            'comparison':'Reuse already saved expanded121 fixed2025-08-01 predictions on same1726 targets; fit213 once',
            'research_candidate':'Second fit on8753 before2026-08-01, not for that held-season evaluation',
            'scope':'Previously examined historical development; no untouched-test or current-accuracy claim',
            'missing':'Raw None retained, existing training-only median +missing indicators; no missing-to-healthy conversion',
            'not_used':['target coach','target lineup','target match minutes','target results as predictors',
                'manager name encoding','news-derived manual strength penalty','current results','LLM forecasts','market'],
            'limitations':['FPL and schedule mostly reused information; only their joint representation plus lagged coach context is new here',
                'Coach labels are past observations, not confirmation of who coaches the target match or exact appointment date',
                'Historical source first-publication and revisions unknown; FPL commit dates are retrospective proxies',
                'Context completeness varies by league and season; no target is dropped for missing context',
                'Old seasons are heavily time-decayed; more columns and rows do not guarantee predictive information',
                'Adding blocks changes tree split opportunities; this is not a causal decomposition'],
            'stop':'Finish the2fixedfits and report; no post-result grid or per-league rescue; no live replacement'})
    rows,coverage = prepare()
    write(OUT/'整合训练输入.json.gz',{'source_files':{k:str(v) for k,v in FILES.items()},'rows':rows})
    write(OUT/'整合覆盖.json',coverage)
    held = [r for r in rows if r['period']=='2025-26']
    if len(held) != 1726:
        raise ValueError('Held cohort changed')
    learner,action = fit(rows,HELD_CUTOFF,'共同情境_20250801')
    current = predictions(learner,held)
    earlier = read(BASE/'扩样_20250801_2025-26预测.json')
    if [r['match_id'] for r in earlier] != [r['match_id'] for r in current]:
        raise ValueError('Comparison must use identical fixtures and ordering')
    if set(learner.parameters()['reference_match_ids']) & {r['match_id'] for r in held}:
        raise ValueError('Evaluation targets entered the meta learner training')
    old_scores,_ = score_details(earlier)
    new_scores,_ = score_details(current)
    write(OUT/'共同情境_2025-26逐场预测.json',current)
    final,final_action = fit(rows,FINAL_CUTOFF,'共同情境_20260801_研究候选')
    restored = joblib.load(OUT/'共同情境_20260801_研究候选.joblib')
    examples = [r['side_features'] for r in rows[-5:]]
    dates = [FINAL_CUTOFF]*5
    replay = max(abs(a-b) for left,right in zip(final.predict_many(examples,dates),
        restored.predict_many(examples,dates),strict=True) for a,b in zip(left,right,strict=True))
    result = {'finished_at_utc':datetime.now(timezone.utc).isoformat(),'feature_n':213,
        'cohort_n':len(rows),'training_20250801_n':learner.parameters()['n_matches'],'target_n':len(held),
        'new_fits_this_execution':int(action=='fitted')+int(final_action=='fitted'),
        'new_base_fits':0,'parameter_searches':0,'coverage':coverage,
        'old_expanded121':old_scores,'joint_context213':new_scores,
        'target_ids_excluded_from_scoring_model_training':True,'original_121_values_preserved':True,
        'final_save_reload_max_rate_difference':replay,
        'final_candidate_used_for_held_scoring':False,'historical_first_publication_reconstructed':False,
        'new_historical_llm_forecasts':0,'adopted_to_live':False,'live_forecasts_changed':False,
        'elapsed_seconds':time.perf_counter()-start,
        'interpretation':'Joint information learning on examined history; no claim that small gains/losses establish contextual causal value'}
    write(OUT/'共同接入训练结果.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

"""Connect the new informed LLM to the existing prospective scorer."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.domain import Fixture
from football_model.multiline import multi_record, score_multi

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/协作预测与大样本方案_20261028'
SOURCE = ROOT/'data/serie_a_collaborative_20261028/情境协作LLM回答.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n',encoding='utf-8')


def main():
    destination = OUT/'情境协作LLM前瞻记录.json'
    if destination.exists():
        print('Existing prospective record retained; no new version or forecast generated.')
        return
    source = read(ROOT/'outputs/今日意甲预测_20261028/前瞻记录_goal_simple.json')
    llm = read(SOURCE)
    if (llm['exposure']['current_statistical_probabilities'] is not True or
            any(llm['exposure'][k] is not False for k in ('market', 'other_llm', 'results'))):
        raise ValueError('This record requires the declared informed input policy')
    answers = {r['fixture_id']:r for r in llm['records']}
    if set(answers) != {r['fixture']['match_id'] for r in source['records']}:
        raise ValueError('Forecasts must cover exactly the same five fixtures')
    now = datetime.now(timezone.utc).isoformat()
    records = []
    for row in source['records']:
        f = row['fixture']
        answer = answers[f['match_id']]
        af = answer['fixture']
        if (af['home'], af['away'], af['kickoff_utc']) != (f['home'], f['away'], f['kickoff']):
            raise ValueError('LLM fixture identity differs from the saved original')
        fixture = Fixture(**(f | {'kickoff':datetime.fromisoformat(f['kickoff'])}))
        track = {'fixture_id': f['match_id'], 'id':f'contextual_llm:{f["match_id"]}:{llm["generated_at_utc"]}',
            'generated_at':llm['generated_at_utc'],
            'probabilities':[answer['probabilities'][k] for k in ('home_win', 'draw', 'away_win')],
            'model':f'Current probabilities + opponent-adjusted context + official evidence; model_id={llm["model_id"]}; uncalibrated subjective LLM',
            'exposed_to_statistical':True, 'exposed_to_market':False}
        r = multi_record(fixture, now, tracks={'statistical':row['tracks']['statistical'],
            'llm_independent':row['tracks']['llm_independent'], 'llm_model_informed':track},
            missing_reasons={'market':'Not used in this contextual analysis; original quotes remain in the original record',
                'joint_pool':'No fitted weight for the new contextual policy; assistant synthesis is qualitative'})
        r['llm_response_source'] = str(SOURCE)
        r['llm_fact_packet'] = llm['input_packet']
        r['llm_assistance_scope'] = llm['exposure']
        records.append(r)
    document = {'assembled_at':now, 'records':records, 'results':{},
        'same_five_fixture_new_versions':5, 'new_unique_fixtures':0,
        'assistant_synthesis':str(OUT/'本轮五场综合研判.json'),
        'original_forecasts_preserved':True, 'learned_weight_for_new_contextual_policy':None}
    write(destination, document)
    write(OUT/'情境协作LLM初始评分.json', score_multi(records,{}))
    print(json.dumps({'path':str(destination),'records':len(records),'settled_results':0},ensure_ascii=False))


if __name__ == '__main__':
    main()

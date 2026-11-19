"""Record the new statistical variant with unchanged, earlier LLM/market lines."""
from datetime import datetime, timezone
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import json

import numpy as np

from football_model.domain import Fixture
from football_model.multiline import multi_record, score_multi

ROOT = REPOSITORY_ROOT
OUT = ROOT / 'outputs/同日赛前模型一致化_20261027'
OLD = ROOT / 'outputs/今日意甲预测_20261027'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    current = read(OUT / '意甲同日新版本.json')
    earlier = read(OLD / '前瞻记录_goal_simple.json')
    old_stats = read(OLD / '统计预测与比分分布.json')
    old_by_id = {r['fixture']['match_id']: r for r in earlier['records']}
    old_s_by_id = {r['fixture']['match_id']: r for r in old_stats['records']}
    assembled = datetime.now(timezone.utc).isoformat()
    records, changes = [], []
    for row in current['records']:
        f = row['fixture']
        fid = f['match_id']
        fields = {k: f[k] for k in ('match_id', 'home', 'away', 'competition', 'neutral')}
        fields['kickoff'] = datetime.fromisoformat(f['kickoff'])
        fixture = Fixture(**fields)
        old = old_by_id[fid]
        for key in ('home', 'away', 'kickoff'):
            if old['fixture'][key] != f[key]:
                raise ValueError('Original/new fixture identity differs')
        if row['actual_score'] is not None or row['actual_outcome'] is not None:
            raise ValueError('A live forecast contains a target result')
        probabilities = np.asarray(row['model']['probabilities'])
        grid = np.asarray(row['model']['score_grid']['probabilities'])
        from_grid = np.asarray([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
        if (not np.isfinite(grid).all() or (grid < 0).any()
                or not np.isclose(grid.sum(), 1, rtol=0, atol=1e-12)
                or not np.allclose(from_grid, probabilities, rtol=0, atol=1e-12)):
            raise ValueError('New H/D/A probabilities differ from score-grid regions')
        stat = {'fixture_id': fid, 'id': f'serie_a_same_day_aligned:{fid}:{current["generated_at"]}',
                'generated_at': current['generated_at'], 'probabilities': probabilities.tolist(),
                'model': 'same-day-trained121ExtraTrees;fixed120trees/depth6/leaf60;28day base parameter cadence'}
        record = multi_record(fixture, assembled, tracks={
            'statistical': stat, 'llm_independent': old['tracks']['llm_independent'],
            'market': old['tracks']['market']}, missing_reasons={
                'llm_model_informed': 'No new model-informed generation.',
                'joint_pool': 'Separate predictions; no numerical pooling.'})
        record.update(source_statistics=str(OUT / '意甲同日新版本.json'),
                      source_previous_version=str(OLD / '前瞻记录_goal_simple.json'),
                      llm_reused_without_regeneration=True, market_reused_without_refresh=True,
                      current_results_queried=False)
        for key in ('llm_independent', 'market'):
            if record['tracks'][key] != old['tracks'][key]:
                raise ValueError('Reused route was changed or given a new timestamp')
        records.append(record)
        prior = old_s_by_id[fid]
        changes.append({'fixture_id': fid,
                        'original_goal': prior['goal_simple']['probabilities'],
                        'original_rich_lead3_transfer': prior['rich_121_research']['probabilities'],
                        'new_same_day_aligned': probabilities.tolist(),
                        'original_llm_unchanged': old['tracks']['llm_independent']['probabilities'],
                        'change_vs_original_rich_pp': (100 * (probabilities - np.asarray(prior['rich_121_research']['probabilities']))).tolist(),
                        'top1_changed_vs_original_rich': int(probabilities.argmax()) != int(np.argmax(prior['rich_121_research']['probabilities']))})
    scores = score_multi(records, {})
    if scores['n_records'] != 3 or scores['n_settled'] != 0:
        raise ValueError('Only the same three unstarted fixtures belong to this version')
    write(OUT / '前瞻记录_同日一致化.json', {
        'pilot_id': 'serie-a-today-20261027-same-day-aligned', 'assembled_at': assembled,
        'records': records, 'results': {}, 'observation_unit': 'same three Serie A fixtures, new statistical variant only'})
    write(OUT / '初始评分.json', scores)
    write(OUT / '本轮接入检查.json', {
        'assembled_at': assembled, 'n_fixtures': 3, 'new_unique_fixtures': 0, 'new_llm_generations': 0,
        'old_predictions_overwritten': False, 'score_grid_HDA_consistent': True,
        'llm_and_market_probabilities_and_timestamps_unchanged': True,
        'n_settled': 0, 'changes': changes})
    print(json.dumps({'records': len(records), 'settled': scores['n_settled'],
                     'top1_changes': sum(r['top1_changed_vs_original_rich'] for r in changes)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

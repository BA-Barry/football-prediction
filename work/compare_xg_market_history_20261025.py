"""Score existing goal/xG forecasts against archived 1X2 prices, with no fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from football_model.evaluation import probability_scores
from football_model.football_data import load_directory, MARKET_COLUMNS
from football_model.real_backtest import de_vig
from football_model.forecast_diagnostics import moving_block_means

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
EARLY = ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z/更早年份复现结果.json'
LATER = ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json'
FD = {p: ROOT/f'data/football-data/{p}/20261022T113804786593Z'
      for p in ('2023-24','2025-26')}
COMPS = ('E0','D1','SP1','I1','F1')
METRICS = ('log_loss','brier_sum','rps','accuracy')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def score_arrays(probabilities, y):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y, dtype=int)
    summary = probability_scores(p,y)
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('This archived cohort needs finite positive probabilities without clipping')
    truth = np.eye(3)[y]
    arrays = {'log_loss': -np.log(p[np.arange(len(y)),y]),
              'brier_sum': np.square(p-truth).sum(axis=1),
              'rps': np.square(np.cumsum(p,axis=1)[:,:2]-np.cumsum(truth,axis=1)[:,:2]).sum(axis=1)/2,
              'accuracy': (p.argmax(axis=1)==y).astype(float)}
    if any(not np.isclose(summary[m],v.mean(),rtol=0,atol=1e-12) for m,v in arrays.items()):
        raise ValueError('Per-fixture and aggregate scoring disagree')
    return summary,arrays


def comparison(rows, market):
    selected = [r for r in rows if market in r['markets']]
    missing = [r['match_id'] for r in rows if market not in r['markets']]
    if not selected:
        return {'n':0, 'missing_match_ids':missing, 'unavailable_reason':'no archived prices, no fallback'}
    y = [r['actual_outcome'] for r in selected]
    p = {arm:[r['model_probabilities'][arm] for r in selected] for arm in ('goal','xg','blend')}
    p['market'] = [r['markets'][market]['probabilities'] for r in selected]
    scores, arrays = {}, {}
    for arm, probabilities in p.items():
        scores[arm], arrays[arm] = score_arrays(probabilities,y)
    order, values = [], []
    for arm,reference in (('goal','market'),('xg','market'),('blend','market'),('blend','goal')):
        for metric in METRICS:
            order.append(f'{arm}_minus_{reference}:{metric}')
            values.append(arrays[arm][metric]-arrays[reference][metric])
    matrix = np.column_stack(values)
    return {'n':len(selected),'full_model_cohort_n':len(rows),'scores':scores,
            'difference_order':order,'mean_differences':dict(zip(order,matrix.mean(axis=0).tolist())),
            'block_intervals':{str(b):moving_block_means([r['date'] for r in selected],matrix,block_weeks=b)
                               for b in (2,4,8)},
            'match_ids':[r['match_id'] for r in selected],'missing_match_ids':missing}


def main():
    now = datetime.now(timezone.utc)
    output = ROOT/'outputs'/now.strftime('进球xG历史市场对照_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    plan = {'created_at':now.isoformat(),'purpose':'evaluate existing candidate against the approved independent archived-market reference',
            'source_experiments':[str(EARLY),str(LATER)],'source_fd_directories':{k:str(v) for k,v in FD.items()},
            'held_target_seasons':['2023-24','2025-26'], 'primary_market':'avg_preclosing',
            'all_markets':list(MARKET_COLUMNS), 'primary':'same-fixture blend minus avg_preclosing 1X2 log loss, each season separately',
            'secondary':['goal/xG versus market','Brier sum','RPS divided by 2','Top1 accuracy','blend minus goal on each market cohort','all other archived quotes'],
            'de_vig':'proportional normalized reciprocal decimal prices, existing project method',
            'coverage':'each quote gets its own same-fixture cohort and missing IDs; no fallback to another quote',
            'budget':{'new_team_fits':0,'new_weights':0,'parameter_grids':0},
            'uncertainty':'joint paired noncircular calendar-week blocks 2/4/8,2000 repeats; each season separately',
            'metric_directions':'negative loss differences better; positive accuracy difference better',
            'timing':'28-day model cutoffs and market quotes need not share information timing; no precise historical quote timestamps',
            'limitations':['historical benchmark, not same-information-time superiority test',
                           'preclosing is neither an asserted opening nor T-24h snapshot; closing only secondary',
                           'already-viewed historical samples and conditional intervals, no full research-selection uncertainty',
                           'probability scoring is not evidence of executable price value or profit'],
            'new_live_results_reviewed':False,'new_model_structure':False,
            'pending_user_choice':'separate home/away-allocation structure remains unexecuted; automatic continuation is not approval',
            'adoption':'diagnostic only, no market fusion, no changes to saved forecasts'}
    write(output/'分析前计划.json',plan)
    earlier = json.loads(EARLY.read_text(encoding='utf-8'))
    later = json.loads(LATER.read_text(encoding='utf-8'))
    tracks = {
        '2023-24':{a:[r for r in earlier['predictions'][a] if r['period']=='2023-24']
                   for a in ('goal','xg','blend')},
        '2025-26':{'goal':[r for r in later['common_goal_predictions'] if r['period']=='2025-26'],
                   'xg':[], 'blend':[r for r in later['common_blend_predictions'] if r['period']=='2025-26']},
    }
    # Independent xG rates are already included in the saved mixture context.
    tracks['2025-26']['xg'] = []
    xg_sources = [ROOT/'outputs/独立xG跨联赛复现_20261024T172741074372Z/跨联赛xG复现结果.json',
                  ROOT/'outputs/独立历史xG前推_20261024T171031805341Z/独立xG前推结果.json']
    for path in xg_sources:
        tracks['2025-26']['xg'].extend(r for r in json.loads(path.read_text(encoding='utf-8'))['new_predictions']
                                      if r['period']=='2025-26')
    fd_reports, observations, seasons, leagues = {}, {}, {}, {}
    for period, local_tracks in tracks.items():
        source,report = load_directory(FD[period],season=period.replace('-','/'))
        if report['rejected_rows'] or report['duplicate_keys']:
            raise ValueError('Historical price source has invalid or duplicate records')
        fd_reports[period] = report
        index = {r.match_id:r for r in source}
        maps = {a:{r['match_id']:r for r in rows} for a,rows in local_tracks.items()}
        if any(len(maps[a]) != len(local_tracks[a]) for a in maps):
            raise ValueError('Duplicate saved model prediction')
        ids = set(maps['goal'])
        if any(set(v)!=ids for v in maps.values()) or len(ids)!=1733:
            raise ValueError('The held same-fixture prediction cohort changed')
        rows = []
        for identifier in sorted(ids,key=lambda k:(maps['goal'][k]['date'],k)):
            match = index[identifier]
            model_rows = {a:v[identifier] for a,v in maps.items()}
            for r in model_rows.values():
                if (r['date'] != match.match_date.isoformat() or r['competition'] != match.competition
                    or (r['home'],r['away']) != (match.home,match.away)
                    or r['actual_score'] != list(match.full_time)
                    or r['actual_outcome'] != (0 if match.full_time[0]>match.full_time[1] else 2 if match.full_time[0]<match.full_time[1] else 1)
                    or match.goal_fit_exclusion is not None):
                    raise ValueError('Historical price and model fixture identities differ')
            markets = {m:{'raw_prices':list(v),'probabilities':de_vig(v).tolist(),
                          'raw_columns':list(MARKET_COLUMNS[m]),'overround':float((1/np.asarray(v)).sum()-1)}
                       for m,v in match.markets.items()}
            # Older predictions stored the actual quote values. Check this source
            # still matches that version rather than silently replacing its odds.
            saved = model_rows['goal']
            if 'market_prices' in saved and ({m:list(v) for m,v in saved['market_prices'].items()}
                                            != {m:list(v) for m,v in match.markets.items()}):
                raise ValueError('Archived price values differ from saved forecast version')
            rows.append({'match_id':identifier,'competition':match.competition,'date':match.match_date.isoformat(),
                         'actual_score':list(match.full_time),'actual_outcome':saved['actual_outcome'],
                         'model_probabilities':{a:r['probabilities'] for a,r in model_rows.items()},
                         'model_fit_before_date':saved['fit_before_date'], 'markets':markets,
                         'source_path':match.source_path,'source_url':match.source_url})
        observations[period] = rows
        seasons[period] = {m:comparison(rows,m) for m in MARKET_COLUMNS}
        leagues[period] = {c:{m:comparison([r for r in rows if r['competition']==c],m) for m in MARKET_COLUMNS}
                            for c in COMPS}
    result = {'plan':plan,'finished_at':datetime.now(timezone.utc).isoformat(),
              'fd_reports':fd_reports,'source_xg_experiments':list(map(str,xg_sources)),
              'seasons':seasons,'by_league':leagues,'observations':observations,
              'new_team_fits':0,'new_weights':0,'parameter_grids':0,'adopted':False}
    write(output/'历史市场对照结果.json',result)
    print(json.dumps({'directory':str(output),'seasons':{p:{m:{'n':v['n'],'scores':v['scores'],
                    'blend_minus_market_LL':v['mean_differences']['blend_minus_market:log_loss']}
                    for m,v in markets.items()} for p,markets in seasons.items()}},ensure_ascii=False))


if __name__ == '__main__':
    main()

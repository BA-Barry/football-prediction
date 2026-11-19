"""Broad early-information model study; fixed architecture families, no grid."""
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn

from football_model.early_history_features import EarlyTeamObservation, EarlyHistoryFeatures
from football_model.early_goal_learners import EarlyGoalLearner
from football_model.forecast_diagnostics import moving_block_means
from football_model.probability import score_grid, summarize
from run_npxg_comparison_20261025 import load_source, pairs, read, COVERAGE, ROOT
from run_shot_increment_20261025 import score_details, write

ARMS = {'glm_basic': ('glm', 'basic'), 'glm_rich': ('glm', 'rich'),
        'hgb_basic': ('hgb', 'basic'), 'hgb_rich': ('hgb', 'rich'),
        'extra_trees_rich': ('extra_trees', 'rich')}
COMPS = ('E0', 'D1', 'SP1', 'I1', 'F1')


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Expected finite nonnegative source measurement')
    return result


def observations(records):
    raw_fixtures = {}
    histories = {}
    for entry in read(COVERAGE)['by_season']:
        payload = read(entry['source'])
        for row in payload['dates']:
            identifier = str(row['id'])
            if identifier in raw_fixtures:
                raise ValueError('Duplicate raw provider match ID')
            raw_fixtures[identifier] = row
        for team_id, team in payload['teams'].items():
            for row in team['history']:
                key = (str(team_id), row['date'], row['h_a'])
                if key in histories:
                    raise ValueError('Duplicate provider team/date/side history')
                histories[key] = row
    result = []
    for record in records.values():
        fixture = raw_fixtures[record.understat_id]
        for i, side in enumerate(('h', 'a')):
            source = histories[(str(fixture[side]['id']), fixture['datetime'], side)]
            result.append(EarlyTeamObservation(
                match_id=record.match_id, match_date=record.match_date,
                team=(record.home, record.away)[i], opponent=(record.away, record.home)[i],
                competition=record.competition, is_home=(side == 'h'),
                goals_for=record.goals[i], goals_against=record.goals[1-i],
                xg_for=record.xg[i], xg_against=record.xg[1-i],
                npxg_for=record.npxg[i], npxg_against=record.npxg[1-i],
                ppda_att=number(source['ppda']['att']), ppda_def=number(source['ppda']['def']),
                ppda_allowed_att=number(source['ppda_allowed']['att']),
                ppda_allowed_def=number(source['ppda_allowed']['def']),
                deep_for=number(source['deep']), deep_against=number(source['deep_allowed'])))
    return result


def base_features(row, side, xg_rates):
    return {**{f'league_{comp}': float(row['competition'] == comp) for comp in COMPS},
        'is_home': float(side == 0), 'log_goal_mean_for': math.log(row['rates'][side]),
        'log_goal_mean_against': math.log(row['rates'][1-side]),
        'log_xg_mean_for': math.log(xg_rates[side]), 'log_xg_mean_against': math.log(xg_rates[1-side])}


def prepare_rows(np_records, independent):
    feature_builder = EarlyHistoryFeatures(observations(np_records))
    all_rows, missing = [], []
    contexts = []
    for name, group in pairs().items():
        xg_map = {r['match_id']: r for r in group['xg']}
        blend_map = {r['match_id']: r for r in group['old_blend']}
        for original in sorted(group['goal'], key=lambda r: (r['date'], r['match_id'])):
            identifier = original['match_id']
            fd = independent[identifier]
            xg = xg_map[identifier]
            if (original['date'] != fd.match_date.isoformat() or original['actual_score'] != list(fd.full_time)
                    or xg['fit_before_date'] != original['fit_before_date'] or xg['home'] != original['home']
                    or xg['away'] != original['away']):
                raise ValueError('Saved fixture/base information mismatch')
            context = feature_builder.for_match(home=original['home'], away=original['away'],
                competition=original['competition'], cutoff=date.fromisoformat(original['fit_before_date']),
                match_date=date.fromisoformat(original['date']))
            contexts.append((original, xg, blend_map[identifier], context, name))
    template = next(c['home_features'] for _, _, _, c, _ in contexts if c['home_features'] is not None)
    rich_keys = sorted(template)
    for row, xg, blend, context, name in contexts:
        basic = [base_features(row, i, xg['rates']) for i in (0, 1)]
        rich = []
        for i, side in enumerate(('home', 'away')):
            data = context[f'{side}_features']
            rich.append({**({k: None for k in rich_keys} if data is None else data), **basic[i],
                         'history_features_available': float(data is not None)})
        if context['home_features'] is None:
            missing.append({'match_id': row['match_id'], 'metadata': context['metadata']})
        h, a = row['actual_score']
        all_rows.append({
            'match_id': row['match_id'], 'date': row['date'], 'period': row['period'],
            'competition': row['competition'], 'home': row['home'], 'away': row['away'],
            'fit_before_date': row['fit_before_date'], 'actual_score': [h, a],
            'actual_outcome': 0 if h > a else 2 if h < a else 1,
            'basic_features': basic, 'rich_features': rich, 'feature_history': context['metadata'],
            'goal_rates': row['rates'], 'xg_rates': xg['rates'], 'old_blend_rates': blend['rates'],
            'base_generation_group': name})
    all_rows.sort(key=lambda r: (r['date'], r['match_id']))
    if len(all_rows) != 6990 or len({r['match_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Original four-season prediction coverage changed')
    return all_rows, missing, rich_keys


def predict_row(context, rates):
    forecast = summarize(score_grid(*rates, rho=0))
    return {**context, 'rates': list(rates), 'forecast': forecast,
            'probabilities': list(forecast['one_x_two'].values())}


def compare(tracks, selected=None, intervals=False):
    ids = [r['match_id'] for r in tracks['goal']]
    if any([r['match_id'] for r in rows] != ids for rows in tracks.values()):
        raise ValueError('Common support required')
    scores, losses = {}, {}
    for arm, rows in tracks.items():
        scores[arm], losses[arm] = score_details(rows)
    contrasts = [('glm_rich', 'glm_basic'), ('hgb_rich', 'hgb_basic'),
                 ('hgb_basic', 'glm_basic'), ('hgb_rich', 'glm_rich'),
                 ('extra_trees_rich', 'hgb_rich')]
    if selected is not None:
        contrasts += [(selected, 'goal'), (selected, 'old_blend')]
    columns, values = [], []
    for arm, reference in contrasts:
        for domain in losses[arm]:
            columns.append(f'{arm}_minus_{reference}:{domain}')
            values.append(losses[arm][domain]-losses[reference][domain])
    matrix = np.column_stack(values)
    return {'n': len(ids), 'scores': scores, 'difference_order': columns,
        'mean_loss_differences': dict(zip(columns, matrix.mean(axis=0).tolist())),
        'block_intervals': {str(b): moving_block_means([r['date'] for r in tracks['goal']], matrix, block_weeks=b)
                            for b in (2, 4, 8)} if intervals else {}, 'match_ids': ids}


def failure_diagnostics(tracks, selected):
    old, new = tracks['goal'], tracks[selected]
    output = {}
    # Defined diagnostic partitions; not used to change models or select subsets.
    for domain in ('all', 'low_total_lt2', 'middle_total_2to3', 'high_total_ge3', 'balanced', 'clear_favourite'):
        indices = []
        for i, row in enumerate(old):
            total, strength = sum(row['rates']), max(row['rates'])/min(row['rates'])
            if (domain == 'all' or domain == 'low_total_lt2' and total < 2
                    or domain == 'middle_total_2to3' and 2 <= total < 3
                    or domain == 'high_total_ge3' and total >= 3
                    or domain == 'balanced' and strength < 1.5
                    or domain == 'clear_favourite' and strength >= 1.5):
                indices.append(i)
        if not indices:
            continue
        original, revised = [old[i] for i in indices], [new[i] for i in indices]
        os, ol = score_details(original)
        ns, nl = score_details(revised)
        output[domain] = {'n': len(indices), 'goal_scores': os, 'selected_scores': ns,
            'differences': {k: float(np.mean(nl[k]-ol[k])) for k in ol},
            'goal_mean_total': float(np.mean([sum(r['rates']) for r in original])),
            'selected_mean_total': float(np.mean([sum(r['rates']) for r in revised])),
            'actual_mean_total': float(np.mean([sum(r['actual_score']) for r in revised]))}
    return output


def main():
    now = datetime.now(timezone.utc)
    out = ROOT/'outputs'/now.strftime('多信息多算法前推_%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    (out/'models').mkdir()
    plan = {'created_at': now.isoformat(), 'purpose': 'broader predictive pipelines with information-by-learner comparison',
        'warm_source_seasons': '2020/21-2021/22', 'initial_model_training': ['2022-23', '2023-24'],
        'chronological_validation': '2024-25', 'held_retrospective_test': '2025-26',
        'arms': ARMS, 'selection': 'minimum 2024/25 complete-score NLL among five declared arms; choose before inspecting later scores',
        'base_features': ['log own/opponent earlier goal and total-xG means', 'is_home', 'five league indicators'],
        'rich_features': ['trailing5/15 goals,xG,NPxG,PPDA components/ratios,deep and conceded counterparts',
            'opponent corresponding summaries', 'recent-minus-medium trends', 'relevant-venue subset summaries',
            'history coverage/staleness/partial league-only schedule counts and gaps'],
        'missing': 'no rich-only target drops; training-only median and explicit missing indicators; entirely unavailable histories flagged',
        'chronology': 'retain each historical example own original28day featurecutoff; new learner uses targets strictly earlier than currentcutoff',
        'update': 'same August1 anchored28day windows; all5leagues jointly modelled; both scoring sides trained together',
        'weighting': '180day half-life on training fixtures; .01 GLM regularization; tree complexity fixed',
        'distribution': 'all learners output fullgoal means into the shared independentPoisson law; not uncertainty mixtures',
        'budget': 'five fixed scheme families, expected22pooledcutoffs=110fits; zero hyperparameter grids or later-season retuning',
        'primary': 'prior-selected scheme versus current goal/total-xG blend full-score NLL in2025/26',
        'secondary': ['1X2 logloss/Brier/Top1', 'total0/1/2/3/4+ logloss', 'away-minus2 logloss',
                      'matched basic/rich and learner contrasts', 'all5league results and descriptive failure partitions'],
        'old_blend': '2024/25 oldblend is in-sample weight and excluded from modelselection/comparison; 2025/26 earlier-trained oldblend is valid context',
        'base_generation': 'saved2022/23-2023/24 goal/xG inputs start2020; saved2024/25-2025/26 inputs start2022; all five arms share same inputs',
        'uncertainty': 'paired2/4/8calendar-week blocks2000repeats; conditional descriptive intervals, no causal claims',
        'source_limits': ['earliercalendar availability assumed; sourcefirstpublication/revisions unknown',
            'partial league-only gaps not allcompetition rest', 'no complete injuries/playeravailability or verified earlyodds timestamps',
            'same domestic league scale only; no claim complete UEFA crossleague model'],
        'lineups_used': False, 'current_results_reviewed': False, 'neural_networks_added': False,
        'new_dependencies': False, 'runtime': {'sklearn': sklearn.__version__}, 'adopted': False}
    write(out/'分析前计划.json', plan)
    np_records, independent, joins, fd_reports, directories = load_source()
    write(out/'数据联结.json', {'source_matches': len(np_records), 'joins': joins,
                              'independent_directories': directories, 'fd_reports': fd_reports})
    rows, missing, rich_keys = prepare_rows(np_records, independent)
    write(out/'历史特征样本.json', {'rows': rows, 'insufficient_histories': missing, 'rich_feature_keys': rich_keys})
    print(json.dumps({'directory': str(out), 'feature_rows': len(rows), 'history_features_per_side': len(rich_keys),
                      'insufficient_histories_retained': len(missing)}, ensure_ascii=False), flush=True)
    tracks = {arm: [] for arm in ARMS}
    tracks.update(goal=[], xg_raw=[], old_blend=[])
    model_parameters, windows = {}, []
    selected = None
    validation = None
    for period in ('2024-25', '2025-26'):
        targets = [r for r in rows if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in targets}):
            start = time.perf_counter()
            held = [r for r in targets if r['fit_before_date'] == cutoff]
            history = [r for r in rows if r['date'] < cutoff]
            fits = {}
            for arm, (kind, features) in ARMS.items():
                train = [{**r, 'side_features': r[f'{features}_features']} for r in history]
                model = EarlyGoalLearner(kind, features).fit(train, date.fromisoformat(cutoff))
                parameters = model.parameters()
                key = f'{arm}:{cutoff}'
                model_parameters[key] = parameters
                joblib.dump(model, out/'models'/f'{arm}_{cutoff}.joblib', compress=3)
                batch_rates = model.predict_many([r[f'{features}_features'] for r in held],
                                                  [date.fromisoformat(r['date']) for r in held])
                for row, rates in zip(held, batch_rates, strict=True):
                    context = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                                  'fit_before_date', 'actual_score', 'actual_outcome')}
                    tracks[arm].append(predict_row(context, rates))
                fits[arm] = {'training_n': len(history), 'prediction_n': len(held), 'parameters_key': key}
            for row in held:
                context = {k: row[k] for k in ('match_id', 'date', 'period', 'competition', 'home', 'away',
                                              'fit_before_date', 'actual_score', 'actual_outcome')}
                for arm, key in (('goal', 'goal_rates'), ('xg_raw', 'xg_rates')):
                    tracks[arm].append(predict_row(context, row[key]))
                if period == '2025-26':
                    tracks['old_blend'].append(predict_row(context, row['old_blend_rates']))
            windows.append({'period': period, 'cutoff': cutoff, 'fits': fits,
                            'elapsed_seconds': time.perf_counter()-start})
            write(out/'训练进度.json', {'windows': windows, 'finished_model_fits': len(model_parameters), 'selected': selected})
            print(json.dumps({'period': period, 'cutoff': cutoff, 'n': len(held),
                              'seconds': windows[-1]['elapsed_seconds']}, ensure_ascii=False), flush=True)
        for arm in tracks:
            tracks[arm].sort(key=lambda r: (r['date'], r['match_id']))
        local = {arm: [r for r in values if r['period'] == period] for arm, values in tracks.items()
                 if arm != 'old_blend' or period == '2025-26'}
        if period == '2024-25':
            validation = compare(local)
            selected = min(ARMS, key=lambda arm: (validation['scores'][arm]['exact_score']['log_loss'], arm))
            write(out/'较早选择.json', {'selected': selected, 'rule': plan['selection'], 'validation': validation})
            print(json.dumps({'validation_selected': selected, 'validation_scores': {
                a: s['exact_score']['log_loss'] for a,s in validation['scores'].items()}}, ensure_ascii=False), flush=True)
        else:
            test = compare(local, selected=selected, intervals=True)
            leagues = {comp: compare({a: [r for r in values if r['competition'] == comp]
                                      for a,values in local.items()}, selected=selected)
                       for comp in COMPS}
            diagnostics = failure_diagnostics(local, selected)
    write(out/'所有模型参数.json', model_parameters)
    write(out/'逐场预测.json', tracks)
    summary = {'plan': plan, 'finished_at': datetime.now(timezone.utc).isoformat(), 'selected': selected,
        'source_matches': len(np_records), 'feature_rows': len(rows), 'history_features_per_side': len(rich_keys),
        'insufficient_histories_retained': missing, 'new_model_fits': len(model_parameters),
        'validation': validation, 'test': test, 'by_league': leagues, 'failure_diagnostics': diagnostics,
        'windows': windows, 'adopted': False}
    write(out/'多方案结果汇总.json', summary)
    print(json.dumps({'directory': str(out), 'fits': len(model_parameters), 'selected': selected,
        'test_scores': test['scores'], 'test_differences': test['mean_loss_differences']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

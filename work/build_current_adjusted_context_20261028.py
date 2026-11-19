"""Reconstruct opponent-adjusted recent xG with strictly earlier-date fits.

This creates a new model-assisted context packet. It never changes an issued
forecast and never labels reconstructed past expectations as old publications.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

import joblib

from football_model.football_data import HistoricalMatch, load_directory
from football_model.xg_history import DateTeamXgModel
from current_rich_model_helpers_20261026 import load_current_rich_sources

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'data/serie_a_today_20261028'
OUT = ROOT/'data/serie_a_adjusted_context_20261028'
CUTOFF = date(2026, 10, 28)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def load_history():
    current, report = load_directory(SOURCE/'raw', season='2026/27')
    if report['rejected_rows'] or report['duplicate_keys']:
        raise ValueError('Independent current fixture source is not resolved')
    current = [replace(r, markets={}) for r in current if r.competition == 'I1']
    known = {(r.match_date, r.home_label, r.away_label):r for r in current}
    raw = SOURCE/'raw/espn_ita1_20261027_completed.json'
    receipt = read(SOURCE/'receipts/espn_ita1_20261027_completed.json')
    aliases = {'AS Roma':'Roma', 'AC Milan':'Milan', 'Internazionale':'Inter'}
    for event in read(raw)['events']:
        contest = event['competitions'][0]
        status = contest.get('status', event.get('status'))['type']
        played = datetime.fromisoformat(event['date'].replace('Z', '+00:00')).date()
        if played >= CUTOFF or status['completed'] is not True or status['name'] not in ('STATUS_FULL_TIME', 'STATUS_FINAL'):
            raise ValueError('Only earlier completed matches supplement current inputs')
        sides = {r['homeAway']:r for r in contest['competitors']}
        labels = [aliases.get(sides[k]['team']['displayName'], sides[k]['team']['displayName']) for k in ('home', 'away')]
        scores = tuple(int(sides[k]['score']) for k in ('home', 'away'))
        key = (played, *labels)
        if key in known:
            if known[key].full_time != scores:
                raise ValueError('Current score sources disagree')
            continue
        home, away = labels
        row = HistoricalMatch(match_id=f'football-data:I1:2026/27:{home}|{away}',
            competition='I1', season='2026/27', match_date=played,
            source_time_text=event['date'], home='football-data:I1:'+home,
            away='football-data:I1:'+away, home_label=home, away_label=away,
            full_time=scores, half_time=None, markets={}, source_path=str(raw),
            source_url=receipt['source_url'], goal_fit_exclusion=None)
        current.append(row)
        known[key] = row
    sources = load_current_rich_sources(SOURCE/'raw/Serie_A_2026.json', current,
        before_date=CUTOFF, season='2026/27', competition='I1')
    history = sorted((r for r in sources['np_records'].values() if r.competition == 'I1' and r.match_date < CUTOFF),
        key=lambda r:(r.match_date,r.match_id))
    if len({r.match_id for r in history}) != len(history):
        raise ValueError('Duplicate history')
    return history, sources['reports']


def plan(history, facts):
    teams = {}
    for fixture in facts['records']:
        for side in ('home', 'away'):
            context = fixture[side]
            team = 'football-data:I1:'+context['model_identity_label']
            rows = [r for r in history if team in (r.home,r.away) and r.match_date.year == 2026
                    and ':2026/27:' in r.match_id][-5:]
            original_ids = [r['source_match_id'] for r in context['recent_league_schedule']]
            if [r.understat_id for r in rows] != original_ids or len(rows) != 5:
                raise ValueError('Preserve exactly the existing recent-five fixture window')
            teams[team] = rows
    matches = {r.match_id:r for rows in teams.values() for r in rows}
    days = sorted({r.match_date for r in matches.values()})
    return teams, matches, days


def obtain_models(history, days):
    models, fits = {}, []
    (OUT/'models').mkdir(parents=True,exist_ok=True)
    for day in days:
        path = OUT/'models'/f'xg_before_{day.isoformat()}.joblib'
        expected = [r.match_id for r in history if r.match_date < day]
        if path.exists():
            model = joblib.load(path)
            new_fit = False
        else:
            model = DateTeamXgModel(.01, 180).fit(history, day)
            joblib.dump(model, path)
            new_fit = True
        parameters = model.parameters()
        if parameters['reference_match_ids'] != expected or parameters['before_date'] != day.isoformat():
            raise ValueError('Cached training ID/date differs from the intended earlier-date prefix')
        if date.fromisoformat(parameters['training_max_date']) >= day:
            raise ValueError('A past-match outcome entered its own expected xG')
        models[day] = model
        fits.append({'before_date':day.isoformat(), 'model_file':str(path), 'new_fit':new_fit,
            'training_n':len(expected), 'training_max_date':parameters['training_max_date'],
            'optimizer_gradient_max_abs':parameters['gradient_max_abs']})
        print(json.dumps({'stage':'past_model_ready', **fits[-1]}), flush=True)
    return models, fits


def team_context(team, rows, models, history):
    items = []
    for row in rows:
        model = models[row.match_date]
        i = 0 if row.home == team else 1
        opponent = (row.away,row.home)[i]
        unknown = sorted({row.home,row.away}-model.teams)
        expected = None if unknown else model.predict(row.home,row.away,'I1',row.match_date)
        xf, xa = row.xg[i], row.xg[1-i]
        ef, ea = (None,None) if expected is None else (expected[i],expected[1-i])
        af = None if ef is None else xf-ef
        da = None if ea is None else xa-ea
        items.append({'match_id':row.match_id, 'understat_id':row.understat_id,
            'date':row.match_date.isoformat(), 'team':team, 'opponent':opponent,
            'is_home':i==0, 'actual_goals_for':row.goals[i], 'actual_goals_against':row.goals[1-i],
            'actual_xg_for':xf, 'actual_xg_against':xa,
            'earlier_expected_xg_for':ef, 'earlier_expected_xg_against':ea,
            'attack_xg_residual':af, 'defense_xga_residual':da,
            'net_xg_residual':None if af is None else af-da,
            'past_fit_before_date':row.match_date.isoformat(),
            'past_training_max_date':model.parameters()['training_max_date'],
            'own_prior_current_season_matches':sum(r.match_date < row.match_date and ':2026/27:' in r.match_id and team in (r.home,r.away) for r in history),
            'opponent_prior_current_season_matches':sum(r.match_date < row.match_date and ':2026/27:' in r.match_id and opponent in (r.home,r.away) for r in history),
            'missing_reason':None if not unknown else 'unknown_team_in_earlier_fit:'+','.join(unknown),
            'source_url':row.source_url})
    available = [r for r in items if r['net_xg_residual'] is not None]
    mean = lambda name: (math.fsum(r[name] for r in items)/5
                         if len(items)==5 and all(r[name] is not None for r in items) else None)
    return {'team':team, 'reference_count':len(rows), 'available_residual_count':len(available),
        'means_require_all_five':True,
        'mean_actual_xg_for':mean('actual_xg_for'), 'mean_actual_xg_against':mean('actual_xg_against'),
        'mean_earlier_expected_xg_for':mean('earlier_expected_xg_for'),
        'mean_earlier_expected_xg_against':mean('earlier_expected_xg_against'),
        'mean_attack_xg_residual':mean('attack_xg_residual'),
        'mean_defense_xga_residual':mean('defense_xga_residual'),
        'mean_net_xg_residual':mean('net_xg_residual'), 'observations':items}


def arithmetic_checks(teams):
    checked = 0
    for context in teams.values():
        for row in context['observations']:
            assert row['past_training_max_date'] < row['past_fit_before_date'] <= row['date'] < CUTOFF.isoformat()
            if row['net_xg_residual'] is None:
                continue
            assert math.isclose(row['attack_xg_residual'],row['actual_xg_for']-row['earlier_expected_xg_for'],abs_tol=1e-12)
            assert math.isclose(row['defense_xga_residual'],row['actual_xg_against']-row['earlier_expected_xg_against'],abs_tol=1e-12)
            assert math.isclose(row['net_xg_residual'],(row['actual_xg_for']-row['actual_xg_against'])-(row['earlier_expected_xg_for']-row['earlier_expected_xg_against']),abs_tol=1e-12)
            checked += 1
        if context['mean_net_xg_residual'] is not None:
            assert math.isclose(context['mean_attack_xg_residual']-context['mean_defense_xga_residual'],context['mean_net_xg_residual'],abs_tol=1e-12)
    return {'team_observations_checked':checked,'training_prefix_checked_for_all_dates':True,
            'residual_sign_and_sum_consistent':True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('inspect','run'),default='inspect')
    args = parser.parse_args()
    facts = read(SOURCE/'facts_only_llm_packet.json')
    history, source_report = load_history()
    team_rows, matches, days = plan(history,facts)
    description = {'stage':'plan', 'history_n':len(history), 'teams':len(team_rows),
        'team_recent_observations':sum(map(len,team_rows.values())), 'unique_past_matches':len(matches),
        'required_date_models':len(days), 'dates':[d.isoformat() for d in days], 'parameter_searches':0}
    print(json.dumps(description),flush=True)
    if args.mode == 'inspect':
        return
    models, fit_records = obtain_models(history,days)
    teams = {t:team_context(t,rows,models,history) for t,rows in team_rows.items()}
    checks = arithmetic_checks(teams)
    generated = datetime.now(timezone.utc).isoformat()
    packet = deepcopy(facts)
    packet['assembled_at'] = generated
    packet['source_descriptive_packet'] = str(SOURCE/'facts_only_llm_packet.json')
    packet['context_mode'] = 'model_assisted_historical_expectations_not_purely_independent_evidence'
    packet['exposure'] = {'current_target_model_probabilities':False, 'market':False,
        'other_llm_probabilities':False, 'target_results':False, 'historical_xg_model_estimates':True}
    packet['opponent_adjustment_method'] = {
        'estimator':'existing DateTeamXgModel regularized attack/opponent-defense/home GLM',
        'regularization':.01, 'half_life_days':180, 'target':'total match xG, not NPxG or goals',
        'attack_residual':'actual xG for minus earlier expected xG for; positive = more chances than expected',
        'defense_residual':'actual xGA minus earlier expected xGA; positive = allowed more chances than expected',
        'net_residual':'attack residual minus defense residual; positive = net chance performance exceeded earlier expectation',
        'means':'unweighted exactly the original five fixtures; no overlapping views counted as independent evidence',
        'generation_interpretation':'Computed now from archived source data with strict earlier calendar-date prefixes; not predictions actually published before these historical fixtures',
    }
    packet['unknowns'] = [v for v in packet['unknowns'] if 'Statistics are descriptive and not opponent-adjusted' not in v]
    packet['unknowns'] += [
        'Original season/last5/venue tables remain raw descriptions; appended residuals are model-derived estimates, not new independent measurements.',
        'Earlier expected xG is reconstructed now; historical source revision and first-publication times unknown.',
        'The estimated earlier opponent strength has uncertainty; no confidence interval or proof that schedule bias is completely removed is supplied.',
        'No control for red cards, penalties, score state, tactics, lineups, manager changes, or full cross-competition workload.',
        'The attack/defense coefficients are pooled across seasons with time decay; this is not a separately learned new-season dynamic state.',
    ]
    packet['instructions'] += (' Historical expected xG comes from statistical models, so label this response model-assisted; do not call agreement with the goal model independent confirmation. Use opponent-adjusted attack/defense and raw observations together, explain disagreements and uncertainty. Do not invent current target probabilities from hidden sources.')
    for fixture in packet['records']:
        if datetime.fromisoformat(fixture['fixture']['kickoff']) <= datetime.fromisoformat(generated):
            raise ValueError('A current target started before this context was generated')
        for side in ('home','away'):
            key = 'football-data:I1:'+fixture[side]['model_identity_label']
            context = teams[key]
            source = fixture[side]['observed_statistics']['last5']
            if not math.isclose(math.fsum(r['actual_xg_for'] for r in context['observations']),source['xG'],abs_tol=1e-5):
                raise ValueError('Adjusted observations differ from the original factual recent xG window')
            fixture[side]['opponent_adjusted_recent_xg'] = context
    write(OUT/'带对手校正的事实包.json',packet)
    write(OUT/'训练与来源记录.json',{'generated_at_utc':generated,'plan':description,'model_fits':fit_records,
        'new_fits_this_run':sum(r['new_fit'] for r in fit_records),'parameter_searches':0,
        'source_report':source_report,'checks':checks,'current_predictions_changed':False,
        'network_requests':0,'limitations':packet['unknowns']})
    lines=['# 当前意甲：对手校正表现接入','',
        f'生成UTC：{generated}。本包对应意大利10月28日五场、北京时间10月29日凌晨。旧预测与旧事实包不变。','',
        f'复用精确联结的{len(history)}场意甲xG历史。10队近5共50个球队观察、{len(matches)}场不同比赛，涉及{len(days)}个日期；每个日期固定设置拟合一次，仅用严格更早日期资料，缓存保存。',
        '这些过去期望是现在重建，不是当时真实发布的赛前记录，不能拿来补作历史LLM前瞻。未知源修订时间也没有还原。','',
        '|球队|实际xG/场|原预期xG/场|进攻残差|防守残差（正=更差）|净残差|可用/5|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for team,context in teams.items():
        fmt=lambda key:'未知' if context[key] is None else f'{context[key]:+.3f}'
        lines.append('|'+team.split(':')[-1]+'|'+'|'.join(fmt(k) for k in ('mean_actual_xg_for','mean_earlier_expected_xg_for','mean_attack_xg_residual','mean_defense_xga_residual','mean_net_xg_residual'))+f'|{context["available_residual_count"]}/5|')
    lines +=['','分量复用已有对手校正攻防研究：实际总xG减当时信息条件下的预期总xG，进攻与防守单列；净残差不是额外独立信号，也不是因果球队能力。均值保留五场完整窗口，若有未知不填0、不改选更老比赛。',
        '',f'本次新增日期级xG拟合{sum(r["new_fit"] for r in fit_records)}次，参数搜索0次、联网0次；50个球队观察逐行验证残差和方向，所有训练ID严格早于相应过去比赛。',
        '', '事实包没有当前目标模型概率、市场赔率、其他LLM预测或目标赛果；但含历史模型估计，因此后续LLM应标模型辅助，不与统计模型重复计独立证据。',
        '', '这里只接入事实解释，不宣称预测准确率改善。红牌、点球、比分状态、伤停、战术、换帅与完整跨赛事负荷未校正，模型的历史对手强度也有误差。',
        '', '后续数值融合的权重应从真实赛前双线档案学习；本包重建的过去期望不作为真实历史LLM记录。','']
    (OUT/'训练与来源说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'stage':'complete','packet':str(OUT/'带对手校正的事实包.json'),
        'new_fits':sum(r['new_fit'] for r in fit_records),'checks':checks},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

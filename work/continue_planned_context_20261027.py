"""Describe known future cup fixtures and archive one facts-only LLM batch.

No model fitting, learned pooling, result retrieval or old forecast replacement.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import math

from football_model.domain import Fixture
from football_model.allcomp_daily_intake import validate_daily_payload
from football_model.planned_schedule import planned_schedule_context
from football_model.multiline import multi_record, score_multi
from refresh_current_all_competition_20261026 import ROOT, read, write
from run_current_rich_forecasts_20261026 import CHINESE

DATA = ROOT/'data/current_all_competition/planned_20261027'
REPAIRED = ROOT/'outputs/全赛事缺口修复与预测_20261027'
OUTPUT = ROOT/'outputs/计划赛程与独立LLM_20261027'
BJ = timezone(timedelta(hours=8))


def build():
    source = read(REPAIRED/'完整对照与输入.json')
    # Only descriptive statistics are projected. Reasoning/probabilities are excluded.
    old_lm = read(ROOT/'work/current_llm_independent_20261026.json')
    descriptions = {r['fixture_id']: r['observed_statistics'] for r in old_lm['records']}
    clubs = read(ROOT/'data/current_all_competition/daily_20261027/schedule_clubs.json')['clubs']
    club_by_label = {r['team'].split(':', 2)[2]: r for r in clubs}
    display, chinese = {}, {}
    for r in source['records']:
        for side in ('home', 'away'):
            club = club_by_label[r[side+'_label']]
            display[club['source_team_id']] = r['fixture'][side]
            chinese[r['fixture'][side]] = CHINESE[r[side+'_label']]
    events, checks = [], []
    for text in ('20261027', '20261028', '20261029'):
        receipt = read(DATA/'receipts'/f'cup_{text}.json')
        if receipt.get('status') != 200:
            raise ValueError('Cup schedule request did not succeed')
        raw = validate_daily_payload(read(DATA/'raw'/f'cup_{text}.json'),
                                     'eng.league_cup', date.fromisoformat(f'{text[:4]}-{text[4:6]}-{text[6:]}'))
        checks.append({'day_utc': text, 'events': len(raw), 'received_at': receipt['received_at'],
                       'source_url': receipt['source_url']})
        for e in raw:
            c = e['competitions'][0]
            sides = {s['homeAway']: s['team'] for s in c['competitors']}
            status = c.get('status', e.get('status', {}))['type']
            if len(c['competitors']) != 2 or set(sides) != {'home', 'away'}:
                raise ValueError('Unknown cup home/away identity')
            events.append({'event_id': str(e['id']), 'competition': 'EFL Cup fourth round',
                'home': display.get(str(sides['home']['id']), sides['home']['displayName']),
                'away': display.get(str(sides['away']['id']), sides['away']['displayName']),
                'kickoff': datetime.fromisoformat(e['date'].replace('Z', '+00:00')).isoformat(),
                'status': 'scheduled' if status['name'] == 'STATUS_SCHEDULED' else status['name'],
                'source_status': status, 'observed_at': receipt['received_at'],
                'source_url': receipt['source_url']})
    # Official PL announcement supplies all eight ties; ESPN supplies exact times.
    expected = {('Fleetwood Town', 'Arsenal'), ('Manchester City', 'Brighton & Hove Albion'),
        ('Everton', 'Newcastle United'), ('Sunderland', 'Brentford'), ('Liverpool', 'Chelsea'),
        ('Bradford City', 'Peterborough United'), ('Fulham', 'Crystal Palace'),
        ('AFC Bournemouth', 'Aston Villa')}
    if {(r['home'], r['away']) for r in events} != expected or len(events) != 8:
        raise ValueError('Daily provider ties differ from the official announcement')
    now = datetime.now(timezone.utc)
    scope = {'competition': 'EFL Cup fourth round', 'checked_utc_dates': ['2026-10-27', '2026-10-28', '2026-10-29'],
        'tie_list_official_source': 'https://www.premierleague.com/en/news/4698490/',
        'tie_list_published_date': '2026-09-16', 'exact_times_source': 'ESPN single-day scoreboards',
        'city_brighton_primary_time_confirmation': 'https://www.mancity.com/news/mens/city-v-brighton-preview-carabao-cup-28-oct-63928427',
        'other_competitions_future_coverage': False, 'checks': checks,
        'meaning': 'An empty team list means no fixture in this checked cup round; it does not prove no other fixtures.'}
    official = read(ROOT/'data/current_all_competition/daily_20261027/raw/official_fpl_fixtures.json')
    confirmations = {r['fixture_id']: r for r in source['confirmed_unstarted_targets']}
    official_by = {r['id']: r for r in official}
    rows = []
    for r in source['records']:
        f = Fixture(**(r['fixture'] | {'kickoff': datetime.fromisoformat(r['fixture']['kickoff'])}))
        confirmation = confirmations[f.match_id]
        current = official_by[confirmation['source_fixture_id']]
        if current.get('started') is not False or f.kickoff <= now or datetime.fromisoformat(current['kickoff_time'].replace('Z', '+00:00')) != f.kickoff:
            raise ValueError('Target source is not unstarted at the declared kickoff')
        histories = {}
        for side in ('home', 'away'):
            h = r['schedule_history']['history'][side]
            club = club_by_label[r[side+'_label']]
            histories[side] = {'team': f.home if side == 'home' else f.away,
                'window_utc': ['2026-09-28', '2026-10-26'],
                'counts': {days: len(v['selected_event_ids']) for days, v in h['windows'].items()},
                'window_usable': {days: v['usable'] for days, v in h['windows'].items()},
                'completed_events': [{k: v for k, v in e.items() if k in ('event_id', 'date_utc_text', 'home_name', 'away_name', 'competition_slug')}
                    for e in club['events'] if e['category'] == 'competitive' and e['completed']],
                'note': 'Observed competition dates/counts, not actual player minutes, fatigue or target-day rest.'}
        rows.append({'fixture': r['fixture'], 'home_chinese': chinese[f.home], 'away_chinese': chinese[f.away],
            'confirmed_unstarted': True, 'official_confirmation': confirmation,
            'observed_statistics': descriptions[f.match_id], 'completed_schedule': histories,
            'planned_schedule': planned_schedule_context(f, events, as_of=now, source_scope=scope)})
    payload = {'assembled_at': now.isoformat(), 'class_order': ['home_win', 'draw', 'away_win'],
        'scope': 'Facts-only next10 packet; one independent subjective LLM judgment, no training.',
        'sources': {'league_statistics': old_lm['input_paths']['current_observed_team_history_and_completed_dates'],
            'previous_season_statistics': old_lm['input_paths']['previous_season_observed_team_history_only'],
            'league_statistics_original_observation_before': old_lm['generated_at'],
            'league_statistics_selection': 'Only archived observed_statistics fields; old reasoning/forecasts excluded.',
            'completed_schedule': str(ROOT/'data/current_all_competition/daily_20261027/schedule_clubs.json'),
            'official_targets': str(ROOT/'data/current_all_competition/daily_20261027/raw/official_fpl_fixtures.json'),
            'official_targets_observed_at': read(ROOT/'data/current_all_competition/daily_20261027/receipts/official_fpl_fixtures.json')['received_at'],
            'planned_cup_scope': scope},
        'unknowns': ['Verified current injuries/player availability and tactical changes are absent.',
            'Upcoming cup outcomes, participants, minutes and rotation are unknown.',
            'Prior-season team comparability after transfers/coaching changes is unknown.',
            'Current league and home/away windows contain only eight and four games; windows overlap.',
            'Opponent strength is not explicitly adjusted in these descriptive totals.',
            'Future competitions outside this cup round have not been comprehensively checked.'],
        'interpretation': ['Kickoff-to-kickoff hours are not actual recovery hours.',
            'No model/market/other-LLM probabilities or subsequent target outcomes are included.',
            'Use current season and longer windows together; do not count overlapping windows as independent observations.',
            'No hand-written mandatory fatigue penalty or assumed rotation probability.'],
        'records': rows}
    write(DATA/'planned_events.json', {'assembled_at': now.isoformat(), 'source_scope': scope, 'events': events})
    write(DATA/'facts_only_llm_packet.json', payload)
    print(json.dumps({'packet': str(DATA/'facts_only_llm_packet.json'), 'targets': len(rows), 'cup_ties': len(events),
        'target_team_cup_participants': sum(bool(r['planned_schedule'][s]['scheduled_between_as_of_and_target']) for r in rows for s in ('home', 'away'))}))


def finalize(llm_path):
    packet = read(DATA/'facts_only_llm_packet.json')
    lm = read(llm_path)
    forecasts = {r['fixture_id']: r for r in lm['records']}
    if len(forecasts) != len(packet['records']) or set(forecasts) != {r['fixture']['match_id'] for r in packet['records']}:
        raise ValueError('LLM response must account for every target once')
    if any(lm['exposure'].get(k) is not False for k in ('statistical_probabilities', 'market_probabilities', 'other_llm_probabilities', 'target_results')):
        raise ValueError('Independent LLM exposure declaration is inconsistent')
    generated = datetime.fromisoformat(lm['generated_at'])
    if generated.tzinfo is None or generated < datetime.fromisoformat(packet['assembled_at']):
        raise ValueError('LLM judgment cannot precede its facts packet')
    saved = read(REPAIRED/'标准前瞻记录.json')
    statistics = {r['fixture']['match_id']: r['tracks']['statistical'] for r in saved['records']}
    old_lm = {r['fixture_id']: r for r in read(ROOT/'work/current_llm_independent_20261026.json')['records']}
    records, comparisons = [], []
    as_of = datetime.now(timezone.utc).isoformat()
    for r in packet['records']:
        f = Fixture(**(r['fixture'] | {'kickoff': datetime.fromisoformat(r['fixture']['kickoff'])}))
        response = forecasts[f.match_id]
        if response['kickoff_utc'] != r['fixture']['kickoff'] or response['home'] != f.home or response['away'] != f.away:
            raise ValueError('LLM fixture differs from the facts packet')
        p = response.get('probabilities')
        track = None
        if p is not None:
            if len(p) != 3 or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1., abs_tol=1e-12):
                raise ValueError('Invalid LLM three-way probabilities')
            track = {'fixture_id': f.match_id, 'id': f'llm_planned_20261027:{f.match_id}:{lm["generated_at"]}',
                'model': f'independent subjective LLM; reported model_id={lm["model_id"]}; planned cup facts',
                'generated_at': lm['generated_at'], 'probabilities': p,
                'exposed_to_statistical': False, 'exposed_to_market': False}
        elif not response.get('abstention_reason'):
            raise ValueError('Missing probability needs an abstention explanation')
        missing = {'market': 'No fresh matching market observation.',
            'llm_model_informed': 'No new informed LLM judgment.',
            'joint_pool': 'No probability pooling; separate judgments retained.'}
        if track is None:
            missing['llm_independent'] = response['abstention_reason']
        record = multi_record(f, as_of, tracks={'statistical': statistics[f.match_id], 'llm_independent': track},
            missing_reasons=missing)
        record.update(statistical_reused_without_change=True,
            statistical_source=str(REPAIRED/'标准前瞻记录.json'), llm_facts_source=str(DATA/'facts_only_llm_packet.json'))
        records.append(record)
        old = old_lm[f.match_id]['probabilities']
        comparisons.append({'fixture': r['fixture'], 'home_chinese': r['home_chinese'], 'away_chinese': r['away_chinese'],
            'old_llm_probabilities': old, 'new_llm_probabilities': p,
            'statistical_probabilities_unchanged': statistics[f.match_id]['probabilities'],
            'llm_delta_pp': [100*(a-b) for a, b in zip(p, old)] if p is not None else None,
            'planned_schedule': r['planned_schedule'], 'judgment': response})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT/'独立LLM原始判断.json', lm)
    write(OUTPUT/'标准前瞻记录.json', {'pilot_id': 'planned-cup-independent-llm-20261027', 'assembled_at': as_of,
        'records': records, 'results': {}, 'scope': 'One new isolated LLM batch; existing statistical forecasts reused with original timestamps.',
        'new_llm_batches': 1, 'new_statistical_forecasts': 0, 'model_fits': 0, 'parameter_searches': 0})
    def leaders(p):
        return {i for i, value in enumerate(p) if value == max(p)}
    available = [r for r in comparisons if r['new_llm_probabilities'] is not None]
    summary = {'new_llm_covered': len(available),
        'old_to_new_llm_top_set_changed': sum(leaders(r['old_llm_probabilities']) != leaders(r['new_llm_probabilities']) for r in available),
        'statistical_and_new_llm_top_set_equal': sum(leaders(r['statistical_probabilities_unchanged']) == leaders(r['new_llm_probabilities']) for r in available),
        'max_old_to_new_llm_single_probability_change_pp': max((abs(v) for r in available for v in r['llm_delta_pp']), default=None),
        'meaning': 'Probability/leading-outcome sensitivity only; not accuracy, calibrated confidence or a causal schedule effect.'}
    write(OUTPUT/'逐场新旧判断.json', {'assembled_at': as_of, 'snapshot_summary': summary, 'records': comparisons,
        'interpretation': 'Changed facts and a fresh LLM generation; differences are not identified causal schedule effects.'})
    write(OUTPUT/'初始评分.json', score_multi(records, {}))
    from continue_prospective_evaluation_20261027 import run
    run(ROOT, OUTPUT/'分组评价', additional_cohorts={
        'next_allcomp_repaired10': str(REPAIRED/'标准前瞻记录.json'),
        'next_planned_cup_llm10': str(OUTPUT/'标准前瞻记录.json')})
    lines = ['# 计划赛程与独立LLM前瞻', '',
        '本轮接入已公布的英国当地10月27—29日（北京时间10月28—30日）联赛杯第四轮计划：8场，其中13支球队出现在下一轮10场联赛中。来源快照中的计划状态均为尚未开始。对阵名单与英超官方公告一致，精确时间取自ESPN单日赛程；曼城—布莱顿时间另获曼城官网核实。', '',
        '[英超官方对阵公告](https://www.premierleague.com/en/news/4698490/)；[曼城官方开球时间](https://www.mancity.com/news/mens/city-v-brighton-preview-carabao-cup-28-oct-63928427)。其他未来赛事没有全面核查，某队本轮杯赛列表为空不等于无其他比赛。', '',
        '## 已知赛程与未知负荷', '', '| 目标比赛 | 主队杯赛→联赛开球间隔 | 客队杯赛→联赛开球间隔 |', '|---|---|---|']
    def gaps(row, side):
        events = row['planned_schedule'][side]['scheduled_between_as_of_and_target']
        return '；'.join(f'{e["home"]}—{e["away"]}，{e["scheduled_kickoff_to_target_hours"]:.2f}小时' for e in events) or '本轮杯赛未见记录'
    for r in comparisons:
        lines.append(f'| {r["home_chinese"]}—{r["away_chinese"]} | {gaps(r, "home")} | {gaps(r, "away")} |')
    lines += ['', '上述是开球到开球间隔，未扣比赛过程，不代表实际恢复时间；没有把多赛一场换成固定扣分，没有假定谁首发、比赛时长或轮换概率。过去28天完赛描述与未来计划分开存放，原统计模型的40项完赛特征不接入未来比赛。', '',
        '## 三项概率', '', f'新LLM判断于北京时间{generated.astimezone(BJ):%Y-%m-%d %H:%M:%S}生成。使用无历史对话的新子任务，只读取白名单事实包；未提供统计／市场／旧LLM概率，也未查询目标赛果。身份按实际可报告信息记录，未冒称某个外部大模型。', '',
        '统计列沿用全赛事缺口修复版本及其原生成时间，未重新训练或改值。目标时间与未开赛状态按官方2026-10-26 16:35:13 UTC快照核对，杯赛快照获取于16:50:45—46 UTC。新旧LLM都是未校准主观概率，整百分点不代表已测得真实胜率。', '',
        '这里“独立”仅指没有看到其他路线的预测概率，不代表错误在统计上相互独立；共用的球队和xG资料可能导致共同误判，不把同向当作两份独立证据。', '',
        '| 北京时间 | 比赛 | 统计主／平／客% | 旧LLM主／平／客% | 新LLM主／平／客% |', '|---|---|---:|---:|---:|']
    pct = lambda p: '／'.join(f'{100*v:.1f}' for v in p) if p is not None else '弃权'
    for r in comparisons:
        kickoff = datetime.fromisoformat(r['fixture']['kickoff']).astimezone(BJ)
        lines.append(f'| {kickoff:%m-%d %H:%M} | {r["home_chinese"]}—{r["away_chinese"]} | {pct(r["statistical_probabilities_unchanged"])} | {pct(r["old_llm_probabilities"])} | {pct(r["new_llm_probabilities"])} |')
    lines += ['', '## 判断依据与反证', '']
    lines += [f'新LLM覆盖{len(available)}/10场，较旧LLM最高概率选项集合改变{summary["old_to_new_llm_top_set_changed"]}场；统计与新LLM最高概率选项集合相同{summary["statistical_and_new_llm_top_set_equal"]}场。相同不自动提高置信度，不同亦不择优掩盖。具体概率、反证和未知因素保留如下。', '']
    if available:
        largest = max(available, key=lambda r: max(map(abs, r['llm_delta_pp'])))
        lines += [f'最大单项变化出现在{largest["home_chinese"]}—{largest["away_chinese"]}：{max(map(abs, largest["llm_delta_pp"])):.1f}个百分点。结合下列杯赛覆盖查看，不能把这批差值全部算作赛程效应；不同信息整理与一次重新生成中的主观权重变化同时存在，本轮没有通过反复生成估计这种波动。', '']
    for r in comparisons:
        j = r['judgment']
        lines += [f'### {r["home_chinese"]}—{r["away_chinese"]}', '',
            '依据：'+'；'.join(j.get('key_facts', [])), '', '反证／不确定：'+'；'.join(j.get('counterarguments', [])), '',
            '赛程解释：'+j.get('schedule_interpretation', '未提供'), '']
    lines += ['## 停止点与下一步', '',
        '数据复核：20队season／last5／home／away及17队previous_season共970个数值与原始Understat重算吻合，最大四舍五入差4e-7；8场杯赛状态／时间、13队参与及全部开球间隔吻合来源。计划赛程边界测试3项、现有多线评分测试70项通过。新组统计记录逐字段等于原统计记录。', '',
        '一次新LLM整批判断已保存，不反复生成至出现喜欢的方向。新训练0、调参0、新统计预测0；新旧版本都保留，10场仍为同10场，不增加独立比赛样本数。初始结算为0，不能声称准确率改善。新事实与一次重新生成同时存在，概率差不能全部归因于杯赛。', '',
        '本轮不平均合并概率。先逐场列分歧及信息缺口，待真实杯赛完成或已核实可用性出现后再追加真实时点的新版本；重大模型结构变更另行讨论。现阶段优先收集可改变判断的新事实，不再围绕同一批历史赛季来回微调。', '',
        '[完整事实包](G:/Codex/football-prediction-2026-10-22/data/current_all_competition/planned_20261027/facts_only_llm_packet.json)；[逐场新旧判断](G:/Codex/football-prediction-2026-10-22/outputs/计划赛程与独立LLM_20261027/逐场新旧判断.json)。']
    (OUTPUT/'计划赛程与独立判断.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'records': len(records), 'new_llm_batches': 1, 'statistical_fits': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'finalize'])
    parser.add_argument('--llm-path', type=Path, default=DATA/'llm_response.json')
    args = parser.parse_args()
    build() if args.action == 'build' else finalize(args.llm_path)

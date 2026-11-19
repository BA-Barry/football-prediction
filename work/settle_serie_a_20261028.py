"""Append a settlement of the saved 27 October Serie A forecast, never refit it.

Raw result downloads and their real observation receipts are separate G: files.
The original prematch forecasts, facts, LLM responses and market snapshot stay
unchanged. No current postmatch market or regenerated LLM enters this scoring.
"""
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re

import numpy as np

from football_model.multiline import score_multi


ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OLD = ROOT/'outputs/今日意甲预测_20261027'
DATA = ROOT/'data/serie_a_settlement_20261028'
OUT = ROOT/'outputs/昨日意甲复核_20261028'
ROUTES = ('goal_simple', 'independent_xg', 'rich_121_research', 'llm_independent', 'market')
LABELS = {'goal_simple': '真实进球主线', 'independent_xg': '独立xG线',
          'rich_121_research': '121项研究线', 'llm_independent': '独立LLM', 'market': '原赛前市场去水'}
OUTCOMES = ('主胜', '平局', '客胜')
NAMES = {'Sassuolo': '萨索洛', 'Lazio': '拉齐奥', 'Roma': '罗马', 'AS Roma': '罗马',
         'Cagliari': '卡利亚里', 'Torino': '都灵', 'Como': '科莫'}
ALIASES = {'AS Roma': 'Roma', 'US Sassuolo': 'Sassuolo', 'Como 1907': 'Como'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def cross_results():
    text = (DATA/'football_italia_results.html').read_text(encoding='utf-8')
    block = re.search(r'<h4>Tuesday 27th\s*<span class="right">October 2026</span></h4>(.*?)</table>', text, re.S)
    if block is None:
        raise ValueError('The downloaded independent page lacks the explicit 27 October 2026 result block')
    result = {}
    for row in re.findall(r'<tr class="home_[^"]+">(.*?)</tr>', block.group(1), re.S):
        home = re.search(r'<td class="team_name team_home".*?<span class="team_name">([^<]+)</span>', row, re.S)
        away = re.search(r'<td class="team_name team_away".*?<span class="team_name">([^<]+)</span>', row, re.S)
        score = re.search(r'<td class="score_time">\s*<span[^>]*>(\d+)</span>\s*-\s*<span[^>]*>(\d+)</span>', row, re.S)
        if not (home and away and score):
            raise ValueError('The independent result row is missing identity or score fields')
        h, a = [ALIASES.get(html.unescape(v), html.unescape(v)) for v in (home.group(1), away.group(1))]
        key = (h, a)
        if key in result:
            raise ValueError('Duplicate independent result identity')
        result[key] = tuple(map(int, score.groups()))
    if len(result) != 3:
        raise ValueError('Expected all three explicit independent result rows')
    return result


def score_probability(p, actual):
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p <= 0) or abs(float(p.sum())-1.) > 1e-12:
        raise ValueError('Keep the exact saved native positive three-probability output')
    tied = np.flatnonzero(p == p.max())
    return {'probabilities': p.tolist(), 'top1_index': int(p.argmax()),
            'top1_outcome': OUTCOMES[int(p.argmax())], 'top1_tied': len(tied) > 1,
            'top1_accuracy': 1./len(tied) if actual in tied else 0.,
            'actual_outcome_probability': float(p[actual]),
            'log_loss': -math.log(float(p[actual])),
            'brier_sum': float(np.square(p-np.eye(3)[actual]).sum())}


def score_distribution(saved, goals):
    grid = np.asarray(saved['score_grid']['probabilities'], dtype=float)
    axes = saved['score_grid']
    home_axis, away_axis = axes['home_goal_axis'], axes['away_goal_axis']
    if grid.shape != (len(home_axis), len(away_axis)) or abs(float(grid.sum())-1) > 1e-12:
        raise ValueError('Original saved score grid is inconsistent')
    h, a = goals
    if h not in home_axis or a not in away_axis:
        raise ValueError('Actual result lies outside the archived supported score grid')
    hi, ai = home_axis.index(h), away_axis.index(a)
    exact_mass = float(grid[hi, ai])
    totals = np.add.outer(home_axis, away_axis)
    n = h+a
    exact_total_mass = float(grid[totals == n].sum())
    high_tail = float(grid[totals >= n].sum())
    expected = float(np.sum(grid*totals))
    return {'expected_total_goals': expected, 'actual_total_goals': n,
            'total_mean_error_actual_minus_predicted': n-expected,
            'actual_score_probability': exact_mass,
            'actual_score_nll': -math.log(exact_mass),
            'actual_exact_total_probability': exact_total_mass,
            'actual_exact_total_nll': -math.log(exact_total_mass),
            'total_at_least_actual_probability': high_tail,
            'probability_total_4_plus': float(saved['summary']['total_goals']['4+']),
            'predicted_over_2_5': float(saved['summary']['over_2_5']),
            'actual_over_2_5': n > 2.5,
            'distribution_source': 'Exact archived score matrix, without refitting or changing its tail'}


def main():
    source_paths = [OLD/'多线概率与价值对照.json', OLD/'统计预测与比分分布.json',
                    *(OLD/f'前瞻记录_{route}.json' for route in ROUTES[:3])]
    originals = {path: path.read_bytes() for path in source_paths}
    comparison = read(source_paths[0])
    distributions = {r['fixture']['match_id']: r for r in read(source_paths[1])['records']}
    receipts = read(DATA/'下载来源.json')['sources']
    receipt = next(row for row in receipts if row['local_path'].endswith('espn_scoreboard_20261027.json'))
    observed = receipt['received_at']
    events = read(DATA/'espn_scoreboard_20261027.json')['events']
    bank = {event['id']: event for event in events}
    independent = cross_results()
    settled, details = {}, []
    for row in comparison['rows']:
        fixture = row['fixture']
        identifier = fixture['match_id']
        event = bank[identifier.rsplit(':', 1)[1]]
        competition = event['competitions'][0]
        status = competition['status']
        if (status['type']['completed'] is not True or status['type']['name'] != 'STATUS_FULL_TIME'
                or status['period'] != 2):
            raise ValueError('Need completed regulation-time second-half results; no extra time or shootout')
        sides = {c['homeAway']: c for c in competition['competitors']}
        home = ALIASES.get(sides['home']['team']['displayName'], sides['home']['team']['displayName'])
        away = ALIASES.get(sides['away']['team']['displayName'], sides['away']['team']['displayName'])
        if (home, away) != tuple(ALIASES.get(fixture[side], fixture[side]) for side in ('home', 'away')):
            raise ValueError('Result identities differ from the original exact event/teams')
        kickoff = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
        if kickoff != datetime.fromisoformat(fixture['kickoff']):
            raise ValueError('Result kickoff differs from the saved forecast fixture')
        goals = tuple(int(sides[side]['score']) for side in ('home', 'away'))
        if independent[(home, away)] != goals:
            raise ValueError('Independent page and primary raw result disagree')
        actual = 0 if goals[0] > goals[1] else 2 if goals[0] < goals[1] else 1
        settled[identifier] = {'fixture_id': identifier, 'settlement': 'regulation_including_stoppage',
            'home_goals': goals[0], 'away_goals': goals[1],
            'finished_by': observed, 'available_at': observed,
            'actual_whistle_time_unknown': True,
            'source_url': receipt['source_url'], 'source_event_id': event['id'],
            'source_status': status,
            'independent_source_url': 'https://football-italia.net/serie-a-results/',
            'independent_match_date_label': 'Tuesday 27th October 2026'}
        probabilities = dict(row['probabilities_by_route']) | {'market': row['market_de_vig']}
        details.append({'fixture': fixture, 'actual_score': list(goals),
            'actual_outcome_index': actual, 'actual_outcome': OUTCOMES[actual],
            'tracks': {route: score_probability(probabilities[route], actual) for route in ROUTES},
            'score_distributions': {route: score_distribution(distributions[identifier][route], goals)
                                    for route in ROUTES[:3]}})
    if len(details) != 3:
        raise ValueError('Keep the complete original three-match cohort')
    OUT.mkdir(parents=True, exist_ok=True)
    scoring = {}
    for route in ROUTES[:3]:
        scores = score_multi(read(OLD/f'前瞻记录_{route}.json')['records'], settled)
        if scores['n_settled'] != 3 or scores['pending_fixture_ids']:
            raise ValueError('The existing settlement scorer did not retain all three targets')
        for record in scores['per_fixture']:
            expected = next(r for r in details if r['fixture']['match_id'] == record['fixture_id'])
            for arm, ours in (('statistical', route), ('llm_independent', 'llm_independent'), ('market', 'market')):
                actual_scores = record['scores'][arm]
                independent_scores = expected['tracks'][ours]
                if (abs(actual_scores['log_loss']-independent_scores['log_loss']) > 1e-12
                        or abs(actual_scores['brier']-independent_scores['brier_sum']) > 1e-12
                        or actual_scores['top1_accuracy'] != independent_scores['top1_accuracy']):
                    raise ValueError('Existing scorer and independent direct calculation differ')
        scoring[route] = scores
        write(OUT/f'多线结算_{route}.json', scores)
    metrics = {}
    for route in ROUTES:
        values = [r['tracks'][route] for r in details]
        metrics[route] = {'n': len(values),
            'top1_correct': math.fsum(r['top1_accuracy'] for r in values),
            'top1_accuracy': math.fsum(r['top1_accuracy'] for r in values)/len(values),
            'mean_log_loss': math.fsum(r['log_loss'] for r in values)/len(values),
            'mean_brier_sum': math.fsum(r['brier_sum'] for r in values)/len(values),
            'top1_tie_count': sum(r['top1_tied'] for r in values)}
    old_unchanged = all(path.read_bytes() == value for path, value in originals.items())
    if not old_unchanged:
        raise ValueError('A previous prediction file changed during settlement')
    result = {'settled_at': datetime.now(timezone.utc).isoformat(),
        'class_order': ['home', 'draw', 'away'],
        'original_prediction_directory': str(OLD), 'new_result_directory': str(DATA),
        'original_primary_route': comparison['primary_for_record'],
        'settlement': 'regulation_including_stoppage', 'match_count': len(details),
        'result_sources': receipts, 'results': settled, 'records': details, 'metrics': metrics,
        'brier_definition': 'sum across the three class squared errors, not divided by three',
        'old_prediction_bytes_unchanged': old_unchanged,
        'new_model_fits': 0, 'new_llm_forecasts': 0, 'parameter_changes': 0,
        'limits': [
            'Three games are three observations, not fifteen independent observations across five routes.',
            'The independent LLM was recorded before kickoff without model or market exposure; identities and disclosure are from the original record.',
            'All directions correct does not establish 100 percent future accuracy or calibrated probabilities.',
            'Score and total-goal distributions underestimated this batch, but three high totals alone do not establish a systematic tail defect or justify increasing a parameter.',
            'The source completion observation is an upper bound finished_by, not an invented exact final-whistle time.',
            'No postmatch explanation is written into the old prematch inputs, and no old odds are replaced by postmatch prices.',
            'No numerical joint model forecast had been recorded; a new postmatch combination must not be called a prior joint hit.']}
    write(OUT/'结算.json', result)
    lines = ['# 昨日意甲三场：原预测结算', '',
        '意大利／UTC日期为2026年10月27日，北京时间为10月28日凌晨。只结算原来记录的三场及其原始概率，不重算预测、不改旧记录、不据三场调参。', '',
        '[ESPN原始赛果接口](https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates=20261027&limit=1000)与[Football Italia结果页](https://football-italia.net/serie-a-results/)逐场交叉一致。ESPN均标记常规时间完场、period=2；实际终场秒数未知，使用本次收到完场资料的时间作为finished_by。', '',
        '|比赛|实际90分钟比分|实际胜平负|', '|---|---:|---|']
    for record in details:
        f = record['fixture']
        lines.append(f"|{NAMES[f['home']]}—{NAMES[f['away']]}|{'-'.join(map(str, record['actual_score']))}|{record['actual_outcome']}|")
    lines += ['', '## 原始胜平负概率与逐场评分', '',
        '三概率依次为主胜／平局／客胜；LL与Brier越低越好。Brier为三类平方误差之和。', '',
        '|比赛|原线路|原主／平／客%|原单选|命中|LL|Brier|', '|---|---|---:|---|---:|---:|---:|']
    for record in details:
        f = record['fixture']; label = f"{NAMES[f['home']]}—{NAMES[f['away']]}"
        for route in ROUTES:
            m = record['tracks'][route]; p = '／'.join(f'{100*v:.2f}' for v in m['probabilities'])
            lines.append(f"|{label}|{LABELS[route]}|{p}|{m['top1_outcome']}|{m['top1_accuracy']:.0f}|{m['log_loss']:.6f}|{m['brier_sum']:.6f}|")
    lines += ['', '|线路|单选正确|平均LL↓|平均Brier↓|', '|---|---:|---:|---:|']
    for route, m in metrics.items():
        lines.append(f"|{LABELS[route]}|{m['top1_correct']:.0f}/3|{m['mean_log_loss']:.6f}|{m['mean_brier_sum']:.6f}|")
    lines += ['', '**五条线路均为3/3，但只有3场不同比赛。** 原主线是进球模型，丰富121列仍是研究列；LLM和市场并列评分，没有生成过数值联合概率，不能补称“联合模型三场全中”。共同方向也不是独立证据。', '',
        '## 命中方向不等于比分与总球预测准确', '',
        '|比赛|统计线|原预期总球|实际总球|原实际比分概率%|原P(总球≥实际)%|', '|---|---|---:|---:|---:|---:|']
    for record in details:
        f = record['fixture']; label = f"{NAMES[f['home']]}—{NAMES[f['away']]}"
        for route, m in record['score_distributions'].items():
            lines.append(f"|{label}|{LABELS[route]}|{m['expected_total_goals']:.3f}|{m['actual_total_goals']}|{100*m['actual_score_probability']:.4f}|{100*m['total_at_least_actual_probability']:.4f}|")
    lines += ['', '三场实际总球为8、6、4。方向全部命中，但原统计分布对这批进球数量估计明显偏低，尤其萨索洛—拉齐奥的8球尾部。3场不足以确认长期系统偏差，不能立即把总球率抬高或更换分布。这里保留为以后较大样本的误差诊断，不以赛后比分反写赛前推理。', '',
        '昨日报价是否有价值仍以当时概率及价格判断。热门全部赢球不能追认当时低于估计公平赔率的报价有正期望；本次不把未经执行的组合计算成真实投注收益。', '',
        '原始预测、LLM答复、赛前市场快照和初始评分文件保持不变；新完场下载、来源收据、完整结算及逐场概率都另存G盘。']
    (OUT/'结算.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'match_count': len(details), 'scores': [r['actual_score'] for r in details],
        'metrics': metrics, 'old_files_unchanged': old_unchanged,
        'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

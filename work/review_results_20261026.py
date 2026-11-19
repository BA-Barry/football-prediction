"""Score saved forecasts against newly read completed reports, without refitting."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from statistics import mean

from football_model.multiline import TRACKS, score_multi

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/赛果对比_20261026'
PREVIOUS = ROOT/'outputs/赛果对比_20261025'
FPL = ROOT/'data/prematch-intelligence/20261024_batch/result_observation_20261026T005954635787Z/result_observations.json'
TICKET = ROOT/'outputs/用户六串一分析_20261025/多路线六场记录.json'
FUTURE = ROOT/'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json'
LABELS = {'statistical':'统计模型','llm_independent':'原独立LLM','market':'市场去水',
          'llm_model_informed':'原知情LLM','joint_pool':'原50:50联合'}
OUTCOME = ['主胜', '平', '客胜']
ZH = {'Ipswich Town':'伊普斯维奇','Nottingham Forest':'诺丁汉森林','Aston Villa':'维拉',
      'Manchester City':'曼城','Arsenal':'阿森纳','Everton':'埃弗顿','Coventry City':'考文垂',
      'Fulham':'富勒姆','Chelsea':'切尔西','Tottenham Hotspur':'热刺','Liverpool':'利物浦',
      'Brighton & Hove Albion':'布莱顿','Crystal Palace':'水晶宫','Newcastle United':'纽卡斯尔',
      'Manchester United':'曼联','AFC Bournemouth':'伯恩茅斯','Hull City':'赫尔城',
      'Brentford':'布伦特福德','Sunderland':'桑德兰','Leeds United':'利兹'}

# These factual observations were read before this script was run. Retrieval
# distinctions are preserved: indexed official text is not called direct body.
REPORTS = [
    ('prematch:premier-league:2026-10-25:liverpool:brighton-and-hove-albion', [2,0],
     'https://www.liverpoolfc.com/news/match-report-brighton-liverpool-premier-league',
     'official_club_indexed_report', 'Official indexed report describes a completed Liverpool win; direct page could not be displayed.'),
    ('prematch:premier-league:2026-10-25:crystal-palace:newcastle-united', [0,2],
     'https://www.cpfc.co.uk/news/match-reports/match-report-highlights-crystal-palace-newcastle-october-2026/',
     'official_club_direct_body', 'Club report includes FT and describes the Newcastle victory.'),
    ('prematch:premier-league:2026-10-25:manchester-united:bournemouth', [0,3],
     'https://www.manutd.com/en/news/report-united-0-bournemouth-3-on-25-october-2026',
     'official_club_indexed_report', 'Full indexed club report describes the completed defeat; direct report page exposed a shell.'),
    ('prematch:premier-league:2026-10-25:hull-city:brentford', [1,1],
     'https://www.brentfordfc.com/en/news/match-reports-hull-city-1-brentford-1-premier-league-25-10-2026',
     'official_club_direct_body', 'Club report describes the completed away draw and a point earned.'),
    ('prematch:premier-league:2026-10-25:sunderland:leeds-united', [0,0],
     'https://www.leedsunited.com/en/news/stormy-stalemate-watch-highlights-from-leeds-0-0-draw-with-sunderland',
     'official_club_direct_body', 'Dated post-match club highlights article describes a completed goalless draw.'),
    ('ticket:7022', [1,3],
     'https://www.bundesliga.com/en/bundesliga/news/sport-club-freiburg-bayern-munich-match-report-highlights-matchday-7-gnabry-kane-olise-39729',
     'official_league_direct_body', 'League report gives the completed Bayern victory and late third goal.'),
    ('ticket:7024', [2,1], 'https://www.realsociedad.eus/en/news/details/plus-three-more',
     'official_club_direct_body', 'Dated club report describes the home victory and its lead surviving the final whistle.'),
    ('ticket:7026', [0,2], 'https://www.udinese.it/news/squadra/udinese-milan-0-2',
     'official_club_direct_body', 'Dated club report describes Milan taking all three points after scoring in each half.'),
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def outcome(score):
    return 0 if score[0]>score[1] else 2 if score[0]<score[1] else 1


def metrics(scores):
    result = {}
    for arm in TRACKS:
        values = [r['scores'][arm] for r in scores['per_fixture'] if arm in r['scores']]
        result[arm] = {'n':len(values), 'top1_hits':sum(r['top1_accuracy'] for r in values),
            'mean_log_loss':mean(r['log_loss'] for r in values) if values else None,
            'mean_brier':mean(r['brier'] for r in values) if values else None}
    return result


def single_metrics(rows, field):
    values = []
    for row in rows:
        p = row[field]
        if p is None:
            continue
        k = row['actual_outcome']
        tops = [i for i,v in enumerate(p) if v == max(p)]
        values.append({'hit': (1/len(tops) if k in tops else 0), 'log_loss':-math.log(p[k]),
            'brier':sum((v-int(i==k))**2 for i,v in enumerate(p))})
    return {'n':len(values), 'top1_hits':sum(r['hit'] for r in values),
        'mean_log_loss':mean(r['log_loss'] for r in values) if values else None,
        'mean_brier':mean(r['brier'] for r in values) if values else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    at = datetime.now(timezone.utc)
    observed = {r['match_id']:r for r in read(FPL)['fixture_observations']}
    ticket = read(TICKET)
    old = read(PREVIOUS/'原10场_原预测与赛果.json')
    fixtures = {r['fixture']['match_id']:r['fixture'] for r in old['records']}
    for row in ticket['records']:
        fixtures['ticket:'+row['id']] = {'match_id':'ticket:'+row['id'], 'kickoff':row['kickoff'], 'name':row['name']}
    additions = {}
    for fid, score, url, basis, evidence in REPORTS:
        fixture = fixtures[fid]
        if datetime.fromisoformat(fixture['kickoff']) >= at:
            raise ValueError('Cannot append a completed observation to a future fixture')
        corroboration = observed.get(fid)
        if corroboration and (not corroboration['verified'] or
                [corroboration['home_goals'],corroboration['away_goals']] != score):
            raise ValueError('Official FPL score or exact identity disagrees with report')
        additions[fid] = {'fixture_id':fid, 'home_goals':score[0], 'away_goals':score[1],
            'available_at':at.isoformat(), 'finished_by':at.isoformat(),
            'settlement':'regulation_including_stoppage', 'source_url':url,
            'completion_basis':basis, 'completion_evidence_summary':evidence,
            'source_match_date':'2026-10-25', 'recorded_after_source_read':True,
            'source_read_time_exact':None, 'completion_time_note':'Current observation upper bound, not exact final whistle',
            'official_FPL_corroboration':corroboration,
            'FPL_limit':'Source flags retained; report supports finality independently of finished/provisional flags.',
            'not_inferred_from_elapsed_time':True}
    write(OUT/'新增完赛观察.json', {'recorded_at':at.isoformat(), 'results':additions,
        'observed_FPL_path':str(FPL), 'prior_results_retained_from':str(PREVIOUS),
        'source_read_note':'Official direct or indexed report text was read with web tools before recording; indexed content is explicitly labelled.'})
    bundles = {}
    for label in ('原10场','球员资料修订9场'):
        before = read(PREVIOUS/(label+'_原预测与赛果.json'))
        after = deepcopy(before)
        case_ids = {r['fixture']['match_id'] for r in after['records']}
        for fid, result in additions.items():
            if fid not in case_ids:
                continue
            if fid in after['results'] and (after['results'][fid]['home_goals'],after['results'][fid]['away_goals']) != (result['home_goals'],result['away_goals']):
                raise ValueError('Previously saved result conflict')
            after['results'].setdefault(fid, result)
        after['scored_at'] = at.isoformat()
        assert before['records'] == after['records']
        scores = score_multi(after['records'], after['results'])
        write(OUT/(label+'_原预测与赛果.json'), after)
        write(OUT/(label+'_评分.json'), scores)
        bundles[label] = {'own':metrics(scores), 'same_five_tracks':scores['all_five_common']}
    case = read(OUT/'原10场_原预测与赛果.json')
    comparisons = []
    for row in case['records']:
        f, tracks = row['fixture'], row['tracks']
        result = case['results'].get(f['match_id'])
        if not result:
            continue
        score = [result['home_goals'],result['away_goals']]
        actual = outcome(score)
        predictions = {}
        for arm, track in tracks.items():
            if track is None:
                predictions[arm] = None
            else:
                p = track['probabilities']; top = max(range(3), key=p.__getitem__)
                predictions[arm] = {'probabilities':p, 'top':OUTCOME[top], 'top_probability':p[top],
                    'actual_probability':p[actual], 'hit':top==actual}
        comparisons.append({'fixture_id':f['match_id'],
            'name':ZH.get(f['home'],f['home'])+'—'+ZH.get(f['away'],f['away']),
            'score':score, 'actual_outcome':actual, 'predictions':predictions, 'source_url':result['source_url']})
    ticket_rows = []
    mapping = {'7013':'prematch:premier-league:2026-10-25:crystal-palace:newcastle-united',
               '7014':'prematch:premier-league:2026-10-25:manchester-united:bournemouth'}
    results = case['results'] | additions
    for original in ticket['records']:
        row = deepcopy(original)
        fid = mapping.get(row['id'], 'ticket:'+row['id'])
        result = results.get(fid)
        row['fixture_id'] = fid
        if result:
            row.update(status='verified_completed', score=[result['home_goals'],result['away_goals']],
                actual_outcome=outcome([result['home_goals'],result['away_goals']]), source_url=result['source_url'])
            row['ticket_selection_hit'] = row['actual_outcome'] in row['selected']
        else:
            row.update(status='no_verified_final_result', score=None, actual_outcome=None, ticket_selection_hit=None)
        ticket_rows.append(row)
    finished_ticket = [r for r in ticket_rows if r['status']=='verified_completed']
    ticket_metrics = {name:single_metrics(finished_ticket, key) for name,key in
        [('statistical','statistical_probabilities'),('llm_market_exposed','llm'),('screenshot_market','market_proportional_devig')]}
    distinct = {r['fixture']['match_id'] for r in case['records']+read(FUTURE)['records']}
    distinct.update(r['fixture_id'] for r in ticket_rows)
    settled = set(case['results']) | {r['fixture_id'] for r in finished_ticket}
    confidence_description = {}
    for arm in ('statistical', 'llm_independent'):
        values = [r['predictions'][arm] for r in comparisons]
        confidence_description[arm] = {'n':len(values),
            'top_probability_below_half':sum(v['top_probability']<.5 for v in values),
            'top_minus_runner_up_below_five_percentage_points':sum(
                sorted(v['probabilities'])[-1]-sorted(v['probabilities'])[-2]<.05 for v in values),
            'mean_top_probability':mean(v['top_probability'] for v in values),
            'meaning':'Description of saved probabilities only; not a fitted exclusion rule or calibrated confidence interval.'}
    summary = {'recorded_at':at.isoformat(), 'recorded_at_beijing':at.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'distinct_recorded_matches':len(distinct), 'settled_distinct_matches':len(settled),
        'pending_distinct_matches':len(distinct-settled), 'original10_completed':len(case['results']),
        'new_completed_distinct_matches':len(additions), 'original_cohorts':bundles,
        'original10_per_fixture':comparisons, 'ticket':ticket_rows, 'ticket_route_metrics':ticket_metrics,
        'saved_probability_description':confidence_description,
        'ticket_selected_hits':sum(r['ticket_selection_hit'] for r in finished_ticket),
        'ticket_completed_legs':len(finished_ticket), 'sixfold_can_still_win':not any(r['ticket_selection_hit'] is False for r in ticket_rows),
        'forecast_records_changed':False, 'model_fit_count':0, 'parameter_changes':0, 'new_llm_forecasts':0,
        'future_oct31_batch_scored':0,
        'limits':['Five completed ticket legs and original10 overlap twice; separate forecast versions are not separate matches.',
            'Ticket LLM saw screenshot odds and is not the old independent LLM route.',
            'Report text supports completion; original FPL flags remain false/provisional and are not silently altered.',
            'Only observed probabilities are scored; missing market and joint lines stay missing.',
            'Small cohort measurements do not establish long-run accuracy or optimization gains.']}
    write(OUT/'赛果与多线路评分汇总.json', summary)
    lines = ['# 已保存预测赛果核对','', '核对时间：'+summary['recorded_at_beijing'], '',
        f"去重{len(distinct)}场，已核实完赛{len(settled)}场，尚待{len(distinct-settled)}场。原英超10场全部完赛；新增8场不同比赛。", '',
        '## 原英超10场：保留首次赛前记录', '',
        '| 比赛 | 赛果 | 统计最高概率结果 | 原独立LLM最高概率结果 |', '|---|---|---|---|']
    for row in comparisons:
        cells=[]
        for arm in ('statistical','llm_independent'):
            p=row['predictions'][arm]
            cells.append(f"{p['top']} {p['top_probability']:.1%} · {'中' if p['hit'] else '未中'}")
        lines.append(f"| [{row['name']}]({row['source_url']}) | {row['score'][0]}–{row['score'][1]} | {' | '.join(cells)} |")
    lines += ['', '| 线路 | 自身覆盖命中 | Log loss | Brier sum |', '|---|---:|---:|---:|']
    for arm,value in bundles['原10场']['own'].items():
        lines.append(f"| {LABELS[arm]} | {value['top1_hits']:g}/{value['n']} | {value['mean_log_loss']:.6f} | {value['mean_brier']:.6f} |")
    lines += ['', '不同覆盖不能直接排名。市场缺桑德兰一场，联合缺伊普斯维奇有效赛前记录。', '',
        '## 全五线共同覆盖', '', '| 线路 | Top1 | Log loss | Brier sum |', '|---|---:|---:|---:|']
    common=bundles['原10场']['same_five_tracks']
    for arm,value in common['metrics'].items():
        lines.append(f"| {LABELS[arm]} | {value['top1_accuracy']:.1%} | {value['mean_log_loss']:.6f} | {value['mean_brier']:.6f} |")
    lines += ['', f"以上全五线共同覆盖{common['n']}场，不把不同分母的命中率直接当作优劣证明。", '',
        '## 用户六串一：票面覆盖与路线预测分别计算', '',
        '| 比赛 | 已核实赛果 | 票面选择 | 票面覆盖 |', '|---|---|---|---|']
    for row in ticket_rows:
        result='尚无已核实完赛结果' if row['score'] is None else f"{row['score'][0]}–{row['score'][1]}"
        hit='待结算' if row['ticket_selection_hit'] is None else '中' if row['ticket_selection_hit'] else '未中'
        name=f"[{row['name']}]({row['source_url']})" if row.get('source_url') else row['name']
        lines.append(f"| {name} | {result} | {'／'.join(OUTCOME[k] for k in row['selected'])} | {hit} |")
    lines += ['', f"票面已结束{len(finished_ticket)}场中覆盖{summary['ticket_selected_hits']}场，水晶宫胜／平落空，六串一已无法全中。LAFC一场尚未核实完赛，未填写结果。", '',
        '票面同批次模型、看过赔率的LLM、票面去水市场各自最高概率命中如下；它们与原10场保留为不同版本，不能赛后挑选。', '',
        '| 线路 | 已完赛同五场命中 | Log loss | Brier sum |','|---|---:|---:|---:|']
    for arm,value in ticket_metrics.items():
        lines.append(f"| {arm} | {value['top1_hits']:g}/{value['n']} | {value['mean_log_loss']:.6f} | {value['mean_brier']:.6f} |")
    lines += ['', '## 失误诊断与后续', '',
        '- 原10场统计与独立LLM各中3场；新增五场只有水晶宫客负被统计排第一，独立LLM只排对利物浦主胜。两个平局都没有被任何原线路排第一。',
        '- 水晶宫一场须区分三项单选排序与票面双选事件：统计客胜38.49%略高于主胜37.67%，但主胜加平合计61.51%；LLM对应为39%和61%。赛前实际建议是“方向可以理解”“水晶宫胜平保留观察”，未明确反对该双选。双选本次落空应如实记录，不能归因为用户选择与预测方向不同；三项单选评分可另行保留，但不能冒充赛前已建议改买纽卡斯尔。',
        '- 曼联一场原统计仍倾主队；票面同批次LLM虽给客胜32%，单项最高仍是主胜40%。买客队不败命中不等于LLM单选客胜命中。',
        '- 原知情LLM重复统计概率，没有形成独立数值增量；原固定50:50联合也未展示稳定优势。后续证据收束解释并非新概率预测，不纳入这次命中统计。',
        f"- 原统计10场中{confidence_description['statistical']['top_probability_below_half']}场最高概率低于50%，{confidence_description['statistical']['top_minus_runner_up_below_five_percentage_points']}场第一与第二选项相差不足5个百分点。最高概率不等于高把握；这些只是已有概率描述，未根据本次输赢设置筛选阈值。",
        '- 三个新完赛欧洲票面方向命中，不能反过来证明此前低赔率有正期望；单场结果无法验证价值判断。',
        '- 比分事实与赛后事件不倒填到赛前输入。对状态滞后、升班马和低估平局的假设继续用较早数据或后续赛前记录检查；不凭这几场重选参数。',
        '- 原9场修订版本已另存评分，作为明确的信息版本对照，未挑成绩更好版本代替首次预测。', '',
        '本轮只追加赛果和重新计算评分。模型、原预测及融合权重保持原值。所有文件在项目目录。']
    (OUT/'赛果核对与优化依据.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('recorded_at_beijing','distinct_recorded_matches','settled_distinct_matches',
        'pending_distinct_matches','original10_completed','ticket_selected_hits','ticket_completed_legs','sixfold_can_still_win')},ensure_ascii=False))
    print(json.dumps({'original10':bundles['原10场']['own'], 'ticket':ticket_metrics},ensure_ascii=False))


if __name__ == '__main__':
    main()

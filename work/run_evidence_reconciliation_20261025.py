"""Explain saved future forecasts with a new explicitly informed LLM review.

No current result requests, new numerical predictions, fitting or pooling.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.evidence_reconciliation import reconcile_evidence

ROOT=REPOSITORY_ROOT
RUN=ROOT/'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z'
OUT=ROOT/'outputs/历史代表性与证据联合_20261025/current'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load():
    records=read(RUN/'batch_case.json')['records']
    packets={r['fixture']['match_id']:r for r in read(RUN/'independent_packets.json')['packets']}
    responses={r['fixture_id']:r for r in read(RUN/'independent_response.json')['forecasts']}
    return records,packets,responses


def compact_packet(packet):
    evidence=[]
    for item in packet.get('evidence',[]):
        row={k:v for k,v in item.items() if k!='source_players'}
        players=item.get('source_players')
        if players is not None:
            row['source_players_with_recorded_minutes']=[r for r in players if r.get('season_minutes',0)>0]
            row['zero_minute_status_counts']=dict(Counter(r['status'] for r in players if r.get('season_minutes',0)==0))
            row['projection_note']='Only flagged/unavailable source rows, not a complete roster. Nonzero source minutes are descriptive, not predicted participation or effects.'
        evidence.append(row)
    return {'fixture':packet['fixture'],'as_of':packet['as_of'],
        'descriptive_context':packet.get('descriptive_context',[]),
        'evidence':evidence,'unknowns':packet.get('unknowns',[])}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    records,packets,responses=load()
    now=datetime.now(timezone.utc)
    cases=[]
    for record in records:
        f=record['fixture'];fid=f['match_id']
        if datetime.fromisoformat(f['kickoff'])<=now:
            continue
        cases.append({'fixture':f,'original_record_as_of':record['as_of'],
            'statistical':record['tracks'].get('statistical'),
            'original_llm_forecast':responses[fid],
            'original_independent_packet':compact_packet(packets[fid]),
            'statistical_scope':record['tracks'].get('statistical',{}),
            'known_missing_layers':['No verified 2025/26->2026/27 player-level minutes mapping supplied',
                'No fitted current personnel, coach, motivation or fatigue effect is supplied',
                'Current statistical forecast is the saved team-goal line, not the research regression_allcomp',
                'Historical squad overlap can describe past usage changes, but has not established an effect on current odds']})
    payload={'prepared_at':now.isoformat(),'scope':'model-informed evidence review of original 2026-10-24 packets; no market track or current results supplied',
        'probability_order':['home','draw','away'],'review_is_new_forecast':False,
        'input_projection':'nonzero source-minute flagged rows retained; zero-minute rows counted by status; full original packets remain at source path',
        'original_packet_source':str(RUN/'independent_packets.json'),
        'cases':cases}
    write(OUT/'LLM分歧检查输入.json',payload)
    print(json.dumps({'prepared_cases':len(cases),'path':str(OUT/'LLM分歧检查输入.json')},ensure_ascii=False))


def assemble():
    records,packets,responses=load()
    raw_review=read(OUT/'LLM分歧检查回答.json')
    reviews={r['fixture_id']:r for r in raw_review['reviews']}
    if len(reviews)!=len(raw_review['reviews']):
        raise ValueError('Duplicate review fixture')
    now=datetime.now(timezone.utc)
    prepared=read(OUT/'LLM分歧检查输入.json')
    expected={r['fixture']['match_id'] for r in prepared['cases']}
    if set(reviews)!=expected:
        raise ValueError('Review must cover exactly the prepared future cases')
    if any(datetime.fromisoformat(r['generated_at']) < datetime.fromisoformat(prepared['prepared_at'])
           for r in reviews.values()):
        raise ValueError('New review cannot precede the input prepared for this run')
    output=[]
    for r in records:
        fid=r['fixture']['match_id']
        if fid not in expected:
            continue
        result=reconcile_evidence(r,packets[fid],reviews[fid],as_of=now)
        result['original_llm_reasoning']=responses[fid]['reasoning_summary']
        result['original_llm_scenarios']=responses[fid]['key_scenarios']
        output.append(result)
    summary={'analyzed_at':now.isoformat(),'cases':len(output),
        'paired_n':sum(r['comparison'] is not None for r in output),
        'same_unique_top_n':sum(r['comparison'] is not None and r['comparison']['same_unique_top'] for r in output),
        'issue_counts':dict(Counter(i['kind'] for r in output for i in r['issues'])),
        'combined_probability_rows':0,'new_numerical_forecasts':0,'model_fits':0,
        'new_result_requests':0,'existing_forecasts_changed':False,
        'scope':'A new informed explanation, not a new independent prediction or validated predictive gain',
        'case_source':str(RUN/'batch_case.json'),'review_source':str(OUT/'LLM分歧检查回答.json')}
    write(OUT/'逐场证据与分歧.json',{'summary':summary,'cases':output})
    write(OUT/'执行摘要.json',summary)
    lines=['# 已有未来比赛的证据联合判读','',
        '本次重新运行的是模型知情LLM解释，不是新独立概率预测。原统计与独立LLM概率来自2026-10-24，新闻也为当时缓存；当前没有刷新医学/名单/赛程状态，不作新的比赛推荐。',
        '', '| 比赛 | 原统计胜/平/负 | 原独立LLM胜/平/负 | 两线最大差 | 共同首选 |',
        '|---|---|---|---:|---|']
    label={'home':'主胜','draw':'平','away':'客胜'}
    def fmt(track):
        return '缺失' if track is None else '/'.join(f'{100*p:.2f}%' for p in track['probabilities'])
    for r in output:
        c=r['comparison'];f=r['fixture']
        delta='—' if c is None else f"{100*max(abs(v) for v in c['llm_minus_statistical'].values()):.2f}个百分点"
        shared='—' if c is None else '、'.join(label[o] for o in c['common_top_outcomes']) or '无'
        lines.append(f"| {f['home']}—{f['away']} | {fmt(r['tracks']['statistical'])} | {fmt(r['tracks']['llm_independent'])} | {delta} | {shared} |")
    lines+=['','共同首选只是排名一致，不能解读为相互独立印证、胜率过半或联合置信度提高。两线概率范围仅描述已有预测差异，不是统计置信区间。',
        '', '## 各场尚需解释的证据与假设','']
    for r in output:
        lines += [f"### {r['fixture']['home']}—{r['fixture']['away']}",'']
        for issue in r['issues']:
            lines.append(f"- [{issue['kind']}] {issue['summary']} 下一步：{issue['next_question']} 来源ID：{', '.join(issue['evidence_ids'])}。")
        lines.append('')
    lines+=['## 本轮决定','',
        '保留原数值，不计算50:50均值，也不让新评论悄悄成为新预测。当前人员连续性缺少可靠跨季映射，明确留空；历史人员分布测量不能填补这一缺口。',
        '本轮完成一个测量和一个真实十场解释流程后结束；不搜索概率修正、选场阈值或融合权重。来源链接可查证其陈述，但评论对比赛的影响仍是未经验证的推断。',
        '代码可复用。下一次若取得可靠的历史人员对应关系及当时可用时间，可计算具体情景；未建立影响量时，先保留分歧和未知，不设置统一人员连续性扣分。','']
    (OUT/'十场联合判读.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare_only else assemble()

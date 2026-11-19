"""Compare unchanged prospective forecasts with observed completed reports.

Uses the existing probability scorer. Existing official-source normalization is
unchanged; the Coventry observation explicitly discloses its media provenance.
"""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
from statistics import mean

from football_model.multiline import TRACKS, score_multi

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/赛果对比_20261025'
ORIGINAL = ROOT/'outputs/五线赛果评分_20261024T163356973597Z/原赛前记录及新增赛果.json'
REVISION = ROOT/'outputs/五线赛果评分_20261024T163357044590Z/原赛前记录及新增赛果.json'
CURRENT20 = ROOT/'outputs/欧战赛程信息修订_20261024T154941744958Z/当前20场信息版本总表.json'
OBS = ROOT/'data/prematch-intelligence/20261024_batch/result_observation_20261025T134226020273Z/result_observations.json'
LABELS = {'statistical':'统计模型','llm_independent':'独立LLM','market':'市场去水',
          'llm_model_informed':'知情LLM','joint_pool':'50:50联合'}
ZH = {'Ipswich Town':'伊普斯维奇','Nottingham Forest':'诺丁汉森林','Aston Villa':'维拉',
      'Manchester City':'曼城','Arsenal':'阿森纳','Everton':'埃弗顿','Coventry City':'考文垂',
      'Fulham':'富勒姆','Chelsea':'切尔西','Tottenham Hotspur':'热刺'}

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def own(scores):
    result={}
    for arm in TRACKS:
        values=[r['scores'][arm] for r in scores['per_fixture'] if arm in r['scores']]
        result[arm]={'n':len(values),'top1_hits':sum(r['top1_accuracy'] for r in values),
                     'mean_log_loss':mean(r['log_loss'] for r in values),
                     'mean_brier':mean(r['brier'] for r in values)}
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    at=datetime.now(timezone.utc)
    snapshot=read(OBS)
    source_rows={r['match_id']:r for r in snapshot['fixture_observations']}
    additions={}
    descriptions=[
      ('prematch:premier-league:2026-10-24:coventry-city:fulham',[0,1],
       'https://www.skysports.com/football/coventry-vs-fulham/report/559517',
       'completed_named_broadcaster_report_with_official_score_corroboration',
       ['Readable Sky Sports report gives the completed 0–1 result and post-match interviews.',
        'Body describes the resulting league-table positions and the next fixtures.'],
       OUT/'考文垂完赛证据.json'),
      ('prematch:premier-league:2026-10-24:chelsea:tottenham-hotspur',[0,2],
       'https://www.chelseafc.com/en/news/article/match-report-chelsea-0-2-tottenham-hotspur',
       'official_home_club_completed_report',
       ['Chelsea official report describes a second Tottenham goal and the visitors seeing out victory.',
        'The goalscorer list records the 77th-minute own goal and the 87th-minute second goal.'],
       OUT/'切尔西完赛证据.json'),
    ]
    for fid,score,url,basis,body,evidence_path in descriptions:
        observed=source_rows[fid]
        assert observed['verified'] and [observed['home_goals'],observed['away_goals']]==score
        assert datetime.fromisoformat(observed['expected_kickoff'].replace('Z','+00:00'))<at
        evidence={'recorded_at':at.isoformat(),'fixture_id':fid,'home':observed['home'],
          'away':observed['away'],'kickoff':observed['expected_kickoff'],'score':score,
          'source_url':url,'completion_basis':basis,'body_completion_evidence':body,
          'retrieval':'Readable body displayed by web tool during this turn; report read before recorded_at.',
          'source_read_time_exact':None,'read_observation_upper_bound':at.isoformat(),
          'official_FPL_corroboration':observed,
          'FPL_limit':'finished remains false/provisional true; score agreement does not change those flags.',
          'not_inferred_from_elapsed_time':True}
        # Preserve the independently saved Chelsea evidence as well.
        if evidence_path.exists():
            evidence_path=evidence_path.with_name(evidence_path.stem+'_汇总.json')
        write(evidence_path,evidence)
        additions[fid]={'fixture_id':fid,'home_goals':score[0],'away_goals':score[1],
          'finished_by':at.isoformat(),'available_at':at.isoformat(),
          'settlement':'regulation_including_stoppage','source_url':url,
          'completion_basis':basis,'evidence_path':str(evidence_path),
          'source_read_observation_upper_bound':at.isoformat(),
          'completion_time_note':'Observation upper bound, not actual final-whistle time.',
          'corroborating_provider':{'path':str(OBS),**observed}}
    bundles={}
    for label,path in [('原10场',ORIGINAL),('球员资料修订9场',REVISION)]:
        before=read(path); after=deepcopy(before)
        for fid,result in additions.items():
            if fid in after['results']:
                old=after['results'][fid]
                assert (old['home_goals'],old['away_goals'])==(result['home_goals'],result['away_goals'])
            else:
                after['results'][fid]=result
        after['scored_at']=at.isoformat()
        assert before['records']==after['records']
        scores=score_multi(after['records'],after['results'])
        write(OUT/(label+'_原预测与赛果.json'),after)
        write(OUT/(label+'_评分.json'),scores)
        bundles[label]={'source_case':str(path),'scores':scores,'own':own(scores)}
    case=read(OUT/'原10场_原预测与赛果.json')
    scores=bundles['原10场']['scores']
    by_id={r['fixture']['match_id']:r for r in case['records']}
    per=[]
    for row in scores['per_fixture']:
        fid=row['fixture_id'];record=by_id[fid]; f=record['fixture'];result=case['results'][fid]
        predicted={}
        for arm in TRACKS:
            t=record['tracks'][arm]
            if t is None:
                predicted[arm]=None
            else:
                p=t['probabilities'];k=max(range(3),key=p.__getitem__)
                predicted[arm]={'top1':['主胜','平','客胜'][k],
                    'probabilities':p,'top1_probability':p[k],
                    'actual_probability':p[['home','draw','away'].index(row['outcome'])],
                    'hit':row['scores'][arm]['top1_accuracy']}
        per.append({'fixture_id':fid,'name':ZH.get(f['home'],f['home'])+'—'+ZH.get(f['away'],f['away']),
            'kickoff_beijing':datetime.fromisoformat(f['kickoff']).astimezone(timezone(timedelta(hours=8))).isoformat(),
            'score':[result['home_goals'],result['away_goals']],'predictions':predicted,
            'source_url':result['source_url']})
    all20=read(CURRENT20)
    ticket=read(ROOT/'outputs/用户六串一分析_20261025/多路线六场记录.json')
    epl_ticket_ids={'7013':'prematch:premier-league:2026-10-25:crystal-palace:newcastle-united',
                    '7014':'prematch:premier-league:2026-10-25:manchester-united:bournemouth'}
    ids={r['fixture']['match_id'] for r in all20['records']}
    ids.update(epl_ticket_ids.get(r['id'],'ticket:'+r['id']) for r in ticket['records'])
    ticket_status=[{'id':r['id'],'name':r['name'],'kickoff':r['kickoff'],
                   'status':'not_started' if datetime.fromisoformat(r['kickoff'])>at else 'requires_result_observation'}
                  for r in ticket['records']]
    summary={'recorded_at':at.isoformat(),'scope':'Beijing Oct24 through Oct25 observed completed matches',
        'new_completed_matches':2,'original10_settled':len(case['results']),
        'unique20_settled':len(case['results']),'including_ticket_unique':len(ids),
        'including_ticket_pending':len(ids)-len(case['results']),
        'per_fixture':per,'metrics':bundles,'ticket_status':ticket_status,
        'forecast_records_changed':False,'model_fits':0,'parameter_changes':0,'new_llm_predictions':0,
        'score_observation':str(OBS),
        'source_notes':['Four results supported by preserved official club reports; Coventry relies on readable Sky report plus official FPL score corroboration.',
                        'Official FPL final flags remain unchanged and are not used to infer finality from minute 90.',
                        'Distinct forecast versions of one fixture do not increase the number of independent matches.']}
    write(OUT/'赛果对比汇总.json',summary)
    lines=['# 昨天与今天的比赛：原赛前预测对比','',
        '观察时间：'+at.astimezone(timezone(timedelta(hours=8))).isoformat(),'',
        '已核实完赛5场，本轮新增考文垂0–1富勒姆、切尔西0–2热刺；当前票面6场均尚未开赛。',
        '仅追加赛果，不改原预测，不重新训练或调用LLM作赛后预测。','',
        '| 北京开球时间 | 比赛与赛果 | 原统计最高概率 | 原独立LLM最高概率 |',
        '|---|---|---|---|']
    for r in per:
        cells=[]
        for arm in ('statistical','llm_independent'):
            p=r['predictions'][arm]
            cells.append(f"{p['top1']} {p['top1_probability']:.1%} · {'中' if p['hit']==1 else '未中'}")
        lines.append(f"| {r['kickoff_beijing'][5:16].replace('T',' ')} | [{r['name']} {r['score'][0]}–{r['score'][1]}]({r['source_url']}) | {' | '.join(cells)} |")
    lines+=['','## 各自覆盖','', '| 线路 | 命中 | Log loss | Brier |','|---|---:|---:|---:|']
    for arm,r in bundles['原10场']['own'].items():
        lines.append(f"| {LABELS[arm]} | {r['top1_hits']:g}/{r['n']} | {r['mean_log_loss']:.6f} | {r['mean_brier']:.6f} |")
    lines+=['','联合缺伊普斯维奇有效赛前记录，2/4不能与其他线路的2/5直接排名。以下固定相同四场比较；LL与Brier越低越好。','',
        '## 全五线共同四场','', '| 线路 | 命中率 | Log loss | Brier |','|---|---:|---:|---:|']
    common=scores['all_five_common']
    for arm,r in common['metrics'].items():
        lines.append(f"| {LABELS[arm]} | {r['top1_accuracy']:.1%} | {r['mean_log_loss']:.6f} | {r['mean_brier']:.6f} |")
    lines+=['','## 目前能判断什么','',
        '- 统计、独立LLM、市场、知情LLM均为这5场中2场命中（40%）；小样本不能当长期准确率。',
        '- 维拉胜、森林战平、热刺胜均未成为原预测最高概率结果；阿森纳与富勒姆方向命中。',
        '- 切尔西失利含乌龙和红牌。它们是赛中已发生事实，不能回填成赛前已知输入，也不能证明所有预测误差由这些事件造成。',
        '- 热刺胜的原统计概率20.53%、原LLM18%、市场22.31%，本轮应记概率损失；不能把非零概率结果发生直接视为公式错误。',
        '- 知情LLM仍复制统计概率，缺少独立数值增量；固定50:50联合尚未展示稳定优势。',
        '- 球员资料修订版也已单独评分并保留同场结果，不选择赛后表现最好的版本冒充原预测。',
        '- 本轮不因5场结果调整模型、校准或融合权重。','',
        '## 来源与状态','',
        '切尔西使用俱乐部完赛正文。考文垂使用Sky具名赛后全文及FPL比分旁证，明确标为媒体来源；没有将其伪称官方，也没有改动既有仅接收官方来源的标准化函数。',
        'FPL本次五场均仍为finished=false/provisional=true；保留这些原值。完赛证据来自赛后正文，绝非只因分钟数为90或时间已过去。',
        f"此前20场加本票6场，去掉重复的水晶宫和曼联比赛后为{len(ids)}场不同比赛；已结算5场、待{len(ids)-5}场。多个版本不重复计算。",'',
        f"[完整逐场概率与评分]({(OUT/'赛果对比汇总.json').as_posix()})",'']
    (OUT/'昨天今天赛果对比.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'昨天今天赛果对比.md'),'own_metrics':bundles['原10场']['own'],
        'same_four_metrics':common,'per_fixture':per,'unique':len(ids)},ensure_ascii=False))

if __name__=='__main__':
    main()

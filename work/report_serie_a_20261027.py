"""Archive separate Serie A forecast routes and a readable prematch report."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import math

from football_model.domain import Fixture
from football_model.multiline import multi_record,score_multi

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
DATA=ROOT/'data/serie_a_today_20261027'
OUT=ROOT/'outputs/今日意甲预测_20261027'
BJ=timezone(timedelta(hours=8))
ZH={'Sassuolo':'萨索洛','Lazio':'拉齐奥','AS Roma':'罗马','Cagliari':'卡利亚里','Torino':'都灵','Como':'科莫'}
FIELDS=('match_id','home','away','competition','kickoff','neutral')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pct(p):return '／'.join(f'{v*100:.1f}' for v in p)

def main():
    stats=read(OUT/'统计预测与比分分布.json')
    lm=read(DATA/'llm_response.json')
    facts=read(DATA/'facts_only_llm_packet.json')
    quotes=read(DATA/'market_observations.json')
    responses={r['fixture_id']:r for r in lm['records']}
    markets={r['event_id']:r for r in quotes['records']}
    fact_by={r['fixture']['match_id']:r for r in facts['records']}
    if set(responses)!=set(fact_by) or len(responses)!=3:
        raise ValueError('Each future fixture needs exactly one independent LLM response')
    if any(lm['exposure'].get(k) is not False for k in ('statistical_probabilities','market_probabilities','other_llm_probabilities','target_results')):
        raise ValueError('LLM exposure declaration does not describe the independent route')
    if datetime.fromisoformat(lm['generated_at'])<datetime.fromisoformat(facts['assembled_at']):
        raise ValueError('LLM judgment cannot precede the facts packet')
    assembled=datetime.now(timezone.utc).isoformat()
    cohorts={key:[] for key in ('goal_simple','independent_xg','rich_121_research')}
    rows=[]
    for s in stats['records']:
        fixture={k:s['fixture'][k] for k in FIELDS}
        f=Fixture(**(fixture|{'kickoff':datetime.fromisoformat(fixture['kickoff'])}))
        if f.kickoff<=datetime.fromisoformat(assembled):raise ValueError('Target already started')
        answer=responses[f.match_id]
        if (answer['home'],answer['away'],answer['kickoff_utc'])!=(f.home,f.away,fixture['kickoff']):
            raise ValueError('Response and statistical fixture identities differ')
        quote=markets[s['fixture']['source_event_id']]
        odds=quote['decimal_odds']
        if len(odds)!=3 or any(not math.isfinite(v) or v<=1 for v in odds):raise ValueError('Need a full same-provider 1X2 quote')
        inv=[1/v for v in odds];total=sum(inv);market_p=[v/total for v in inv]
        shared={'llm_independent':{'fixture_id':f.match_id,'id':f'serie_a_llm:{f.match_id}:{lm["generated_at"]}',
            'generated_at':lm['generated_at'],'probabilities':answer['probabilities'],
            'model':f'one independent subjective LLM generation; model_id={lm["model_id"]}',
            'exposed_to_statistical':False,'exposed_to_market':False},
            'market':{'fixture_id':f.match_id,'id':f'draftkings_espn:{f.match_id}:{quote["observed_at"]}',
                'observed_at':quote['observed_at'],'probabilities':market_p,'bookmaker':quote['provider'],
                'source':quote['source_url'],'description':'Normalized reciprocal decimal odds; source per-quote update timestamp unknown.'}}
        probability_lines={k:s[k]['probabilities'] for k in cohorts}
        probability_lines['llm_independent']=answer['probabilities']
        for key in cohorts:
            stat={'fixture_id':f.match_id,'id':f'serie_a_{key}:{f.match_id}:{stats["generated_at"]}',
                'generated_at':stats['generated_at'],'probabilities':s[key]['probabilities'],
                'model':{'goal_simple':'I1 regularized Poisson alpha.01 half-life180; fitted earlier results',
                    'independent_xg':'I1 independent xG alpha.01 half-life180; fitted earlier results',
                    'rich_121_research':'saved five-league ExtraTrees121; current lead0 outside training lead3'}[key]}
            record=multi_record(f,assembled,tracks={'statistical':stat,**shared},
                missing_reasons={'llm_model_informed':'No model-informed probability generated.',
                                 'joint_pool':'Separate routes, no numerical pooling.'})
            record.update(source_statistics=str(OUT/'统计预测与比分分布.json'),
                source_llm=str(DATA/'llm_response.json'),source_market=str(DATA/'market_observations.json'))
            cohorts[key].append(record)
        favored_index=0 if f.home=='AS Roma' else 2
        fair_by_line={k:[1/p for p in value] for k,value in probability_lines.items()}
        expected_returns={k:[p*o-1 for p,o in zip(value,odds)] for k,value in probability_lines.items()}
        rows.append({'fixture':fixture,'probabilities_by_route':probability_lines,'market_de_vig':market_p,
            'market_overround':total-1,'quote':quote,'unbuffered_fair_decimal_odds_by_route':fair_by_line,
            'point_estimate_expected_net_return_by_route':expected_returns,
            'favorite_index':favored_index,
            'all_four_routes_favorite_below_price_break_even':all(v[favored_index]<inv[favored_index] for v in probability_lines.values()),
            'factual_context':fact_by[f.match_id],'llm_judgment':answer,
            'goal_distribution':s['goal_simple'],'xg_distribution':s['independent_xg'],
            'rich_distribution':s['rich_121_research'],'actual_outcome':None})
    for key,records in cohorts.items():
        write(OUT/f'前瞻记录_{key}.json',{'pilot_id':f'serie-a-today-20261027-{key}',
            'assembled_at':assembled,'records':records,'results':{},
            'observation_unit':'three fixtures; repeated LLM/market values across model variants are not additional independent matches'})
        write(OUT/f'初始评分_{key}.json',score_multi(records,{}))
    parlay={}
    for route in (*cohorts,'llm_independent'):
        product=math.prod(r['probabilities_by_route'][route][r['favorite_index']] for r in rows)
        price=math.prod(r['quote']['decimal_odds'][r['favorite_index']] for r in rows)
        parlay[route]={'independence_assumption_probability':product,'quote_decimal_product':price,
            'point_estimate_net_return_under_independence':product*price-1,
            'interpretation':'Descriptive cross-match product, not a learned joint model, validated return or recommendation.'}
    write(OUT/'多线概率与价值对照.json',{'assembled_at':assembled,'class_order':['home','draw','away'],
        'unique_fixtures':3,'rows':rows,'favorite_three_fold_descriptive':parlay,
        'combined_probabilities':None,'results':{},'primary_for_record':'goal_simple',
        'primary_choice_reason':'Same-day base goal model needs no lead3-to-lead0 meta-model transfer; rich version retained separately.',
        'new_statistical_fits':2,'new_tree_fits':0,'parameter_searches':0,'llm_batches':1})
    lines=['# 今晚意甲赛前预测', '',
        '本报告按意大利当地2026年10月27日晚的3场比赛生成；北京时间均为10月28日凌晨。北京时间10月27日剩余时段没有这3场开球，日期不混写。[意甲官方赛程](https://www.legaseriea.it/serie-a/news/quando-si-gioca-anticipi-e-posticipi-fino-alla-12a-giornata)。', '',
        f'统计生成：北京时间{datetime.fromisoformat(stats["generated_at"]).astimezone(BJ):%m-%d %H:%M:%S}；独立LLM生成：{datetime.fromisoformat(lm["generated_at"]).astimezone(BJ):%m-%d %H:%M:%S}；市场采集：{datetime.fromisoformat(quotes["observed_at"]).astimezone(BJ):%m-%d %H:%M:%S}。均早于开球，全部新增资料、模型及结果在G盘。', '',
        '## 主胜／平局／客胜', '',
        '三项分别列出，不用双选覆盖代替单项命中。进球主线按现有固定设置重算意甲历史，xG独立对照；丰富研究列复用含意甲训练样本的121项树模型。各线保留，不简单平均。', '',
        '| 北京时间 | 比赛 | 进球主线% | 独立xG% | 丰富研究列% | 独立LLM% | 市场去水% |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        f=r['fixture'];dt=datetime.fromisoformat(f['kickoff']).astimezone(BJ);p=r['probabilities_by_route']
        lines.append(f'| {dt:%m-%d %H:%M} | {ZH[f["home"]]}—{ZH[f["away"]]} | {pct(p["goal_simple"])} | {pct(p["independent_xg"])} | {pct(p["rich_121_research"])} | {pct(p["llm_independent"])} | {pct(r["market_de_vig"])} |')
    lines+=['', '方向：罗马主胜的依据最集中；拉齐奥仅为小幅客胜倾向；科莫客胜方向的统计与LLM差距较大，不能当稳胆。共同方向不代表误差独立，也不自动提高置信度。', '',
        '丰富列的历史训练为提前3个日期日，本次为同日提前若干小时应用，泛化未经验证，故标准主线记录先使用可直接同日运行的进球模型。丰富列仍完整保存，不据这3场挑有利线路；独立LLM也未经经验校准。', '',
        '## 比分与总进球', '',
        '下表来自进球主线的联合比分分布，比分为概率最高的三个具体结果。最大单个比分的概率仅约一成，不当作确定比分；总球2与3分别列出。', '',
        '| 比赛 | 预期进球 主／客 | 三个比分候选 | 0球% | 1球% | 2球% | 3球% | 4+球% | 大于2.5球% |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        f=r['fixture'];d=r['goal_distribution'];total=d['summary']['total_goals'];scores='、'.join(f'{v["score"]}（{100*v["probability"]:.1f}%）' for v in d['top_scores'][:3])
        lines.append(f'| {ZH[f["home"]]}—{ZH[f["away"]]} | {d["rates"][0]:.2f}／{d["rates"][1]:.2f} | {scores} | '+' | '.join(f'{100*total[k]:.1f}' for k in ('0','1','2','3','4+'))+f' | {100*d["over_2_5"]:.1f} |')
    lines+=['', '完整比分网格、0至7+球分布以及各统计路线全部候选保存在统计JSON。现有模型没有半场联合训练，本轮不填造半全场概率。', '',
        '## 方向和价格分开判断', '',
        '[ESPN / DraftKings同源报价](https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates=20261027&limit=1000)。源未提供逐项更新时间，表中只是本次获取快照；open／close是源字段，赛前close不称最终封盘，变动原因和发生时刻未知。', '',
        '| 比赛 | 来源open主／平／客 | 本次报价主／平／客 | 主方向 | 四路线公平赔率跨度 |', '|---|---:|---:|---|---:|']
    for r in rows:
        f=r['fixture'];idx=r['favorite_index'];q=r['quote'];fair=[v[idx] for v in r['unbuffered_fair_decimal_odds_by_route'].values()]
        fmt=lambda x:'／'.join(f'{v:.2f}' for v in x)
        lines.append(f'| {ZH[f["home"]]}—{ZH[f["away"]]} | {fmt(q["opening_odds"])} | {fmt(q["decimal_odds"])} | '+('主胜' if idx==0 else '客胜')+f' | {min(fair):.2f}—{max(fair):.2f} |')
    lines+=['', '公平赔率为各路线1/p的无缓冲盈亏平衡点；跨度只是不同估计之间的差异，不是置信区间或已确认投注门槛。三个热门当前报价均低于四条估计各自的公平赔率，点估计下均没有价格优势。罗马更可能赢，不能据此认可1.31的价格。', '',
        '卡利亚里客胜和都灵主胜的长赔，在部分点估计下出现表面正价差；这些结果本身发生概率低，球队信息不完整、模型与LLM又未校准，不据此升级成强推荐。三热门二串／三串本轮均不列为有价值候选，不为了凑单强行组合。', '',
        '## 逐场依据与反证', '']
    for r in rows:
        f=r['fixture'];a=r['llm_judgment'];context=r['factual_context']
        lines +=[f'### {ZH[f["home"]]}—{ZH[f["away"]]}', '',
            '球队事实：'+'；'.join(a['key_facts'][:3]), '',
            '反证：'+'；'.join(a['counterarguments'][:2]), '',
            '可用性：'+a['availability_interpretation'], '',
            '赛程：'+a['schedule_interpretation'], '',
            '不确定性：'+a['uncertainty'], '']
        for side in ('home','away'):
            news=context[side]['official_availability'];sources={c['source_url']:c.get('source_title','官方消息') for c in news['claims'] if c.get('source_url')}
            if sources:lines.append(ZH[context[side]['team']]+'来源：'+'；'.join(f'[{title}]({url})' for url,title in sources.items()))
            else:lines.append(ZH[context[side]['team']]+'：没有可用的本场官方正文，状态保留未知。')
        lines.append('')
    lines+=['## 数据与执行范围', '',
        '新取得2026/27意甲Football-Data比分与Understat逐场数据各80场，并按日期、对阵和全场比分连接。进球主线使用2360条意甲历史候选，沿用既存特殊比赛处理后实际拟合2358场；独立xG拟合2356场。固定正则0.01、时间半衰期180天，未作参数搜索。历史新下载时点不冒充当时公布时点。', '',
        '121项树直接加载原模型：总训练6967场，其中意甲1493场，训练最晚2026-05-24。本轮没有重训树；同日特征仅用10月27日前的比赛，日期真实，不把北京时间28日改作意大利目标日期以绕过时间限制。同日组装公式已与原较早日期接口逐字段核对726项一致，保存重载预测一致；这些是输入与保存正确性检查，不能替代泛化验证。', '',
        '三场LLM只读无其他路线概率的球队／赛程／官方消息包；未看市场、统计或其他LLM概率，身份无法确知记unknown。独立仅指输入暴露状态，共用事实仍会导致相关错误。未取得本场正文的队伍不当作全员健康，10月10日或21日旧伤情不外推今晚必缺，已持续伤缺不机械重复扣分。', '',
        '本轮3场各自保留3种统计版本，复用的LLM与市场不会算成9场独立比赛；意甲记录与原英超20场分开，不改原预测。新训练为固定进球／xG各1次，新树训练0，调参0，新LLM整批1次；赛果全部留空，首次前瞻评分0场，不声称命中率或盈利已验证。', '',
        '[全部路线概率、价格及来源](G:/Codex/football-prediction-2026-10-22/outputs/今日意甲预测_20261027/多线概率与价值对照.json)；[完整比分分布](G:/Codex/football-prediction-2026-10-22/outputs/今日意甲预测_20261027/统计预测与比分分布.json)；[进球主线前瞻记录](G:/Codex/football-prediction-2026-10-22/outputs/今日意甲预测_20261027/前瞻记录_goal_simple.json)。']
    (OUT/'今晚意甲预测.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT/'今晚意甲预测.md'),'unique_fixtures':3,'statistical_versions':3,'llm_batches':1,'settled':0},ensure_ascii=False))

if __name__=='__main__':main()

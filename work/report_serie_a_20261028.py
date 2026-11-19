"""Record the five current I1 fixtures across distinct statistical/LLM/market lines."""
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path

from football_model.domain import Fixture
from football_model.multiline import multi_record,score_multi

ROOT=Path('G:/Codex/football-prediction-2026-10-22')
DATA=ROOT/'data/serie_a_today_20261028';OUT=ROOT/'outputs/今日意甲预测_20261028'
BJ=timezone(timedelta(hours=8))
ZH={'AC Milan':'AC米兰','Bologna':'博洛尼亚','Parma':'帕尔马','Udinese':'乌迪内斯','Venezia':'威尼斯','Internazionale':'国际米兰','Genoa':'热那亚','Juventus':'尤文图斯','Monza':'蒙扎','Napoli':'那不勒斯'}
ROUTES=('goal_simple','independent_xg','rich_121_research')
LABELS={'goal_simple':'进球主线','independent_xg':'独立xG','rich_121_research':'121项研究','llm_independent':'独立LLM'}
FIELDS=('match_id','home','away','competition','kickoff','neutral')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def time(v):return datetime.fromisoformat(v.replace('Z','+00:00')).astimezone(timezone.utc)
def pct(p):return '／'.join(f'{100*x:.1f}' for x in p)
def price(p):return '／'.join(f'{x:.2f}' for x in p)
def title(f):return ZH[f['home']]+'—'+ZH[f['away']]
def text(v):return '；'.join(map(str,v)) if isinstance(v,list) else str(v)

def main():
    if not (DATA/'llm_response.json').exists():
        raise ValueError('Independent LLM response not yet available; do not fabricate or resample it')
    stats=read(OUT/'统计预测与比分分布.json');lm=read(DATA/'llm_response.json')
    facts=read(DATA/'facts_only_llm_packet.json');quotes=read(DATA/'market_observations.json')
    assembled=datetime.now(timezone.utc);generated=lm['generated_at_utc']
    if any(lm['exposure'].get(k) is not False for k in ('statistics','market','other_llm','target_results')):
        raise ValueError('Independent LLM exposure declaration differs')
    if time(generated)<time(facts['assembled_at']) or time(generated)>assembled:
        raise ValueError('LLM actual generation must follow its packet and precede this report')
    answers={r.get('fixture_id',r.get('fixture',{}).get('match_id')):r for r in lm['records']};context={r['fixture']['match_id']:r for r in facts['records']}
    markets={r['event_id']:r for r in quotes['records']}
    targets={r['fixture']['match_id'] for r in stats['records']}
    if len(targets)!=5 or set(answers)!=targets or set(context)!=targets or len(answers)!=len(lm['records']):
        raise ValueError('Keep all five unique current fixtures and exactly one LLM response each')
    cohorts={k:[] for k in ROUTES};rows=[]
    for s in stats['records']:
        f={k:s['fixture'][k] for k in FIELDS};fixture=Fixture(**(f|{'kickoff':time(f['kickoff'])}))
        if fixture.kickoff<=assembled:raise ValueError('A target has started before report assembly')
        a=answers[f['match_id']]
        if (a['home'],a['away'])!=(f['home'],f['away']):raise ValueError('LLM fixture team identity mismatch')
        if 'kickoff_utc' in a and time(a['kickoff_utc'])!=fixture.kickoff:raise ValueError('LLM kickoff mismatch')
        if 'fixture' in a and a['fixture']!=f:raise ValueError('LLM complete fixture differs')
        p=a['probabilities']
        if len(p)!=3 or any(not math.isfinite(x) or not 0<x<1 for x in p) or abs(sum(p)-1)>1e-12:
            raise ValueError('Need three native positive LLM probabilities summing to one')
        q=markets[s['fixture']['source_event_id']];odds=q['decimal_odds']
        if len(odds)!=3 or any(not math.isfinite(v) or v<=1 for v in odds):raise ValueError('Need same-provider H/D/A quote')
        inv=[1/o for o in odds];overround=sum(inv);market=[v/overround for v in inv]
        shared={'llm_independent':{'fixture_id':f['match_id'],'id':f'llm:{f["match_id"]}:{generated}',
            'generated_at':generated,'probabilities':p,'model':f'Independent subjective LLM; actual model_id={lm["model_id"]}',
            'exposed_to_statistical':False,'exposed_to_market':False},
            'market':{'fixture_id':f['match_id'],'id':f'market:{f["match_id"]}:{q["observed_at"]}',
            'observed_at':q['observed_at'],'probabilities':market,'bookmaker':q['provider'],'source':q['source_url'],
            'description':'Reciprocal decimal odds normalized across one provider; per-quote update timestamp unknown'}}
        route_p={k:s[k]['probabilities'] for k in ROUTES}|{'llm_independent':p}
        for route in ROUTES:
            track={'fixture_id':f['match_id'],'id':f'{route}:{f["match_id"]}:{stats["generated_at"]}',
                'generated_at':stats['generated_at'],'probabilities':route_p[route],
                'model':{'goal_simple':'I1 fixed alpha.01 half-life180 goal model; strictly earlier scores',
                    'independent_xg':'I1 fixed alpha.01 half-life180 xG model; separate research',
                    'rich_121_research':'Saved pooled121 tree; lead3 training applied at lead0 without validated transfer'}[route]}
            rec=multi_record(fixture,assembled,tracks={'statistical':track,**shared},
                missing_reasons={'llm_model_informed':'No model-informed probability generated',
                    'joint_pool':'Routes retained separately; no numerical pooling'})
            rec.update(source_statistics=str(OUT/'统计预测与比分分布.json'),source_llm=str(DATA/'llm_response.json'),
                source_market=str(DATA/'market_observations.json'))
            cohorts[route].append(rec)
        favorite=max(range(3),key=market.__getitem__)
        rows.append({'fixture':f,'probabilities_by_route':route_p,'market_de_vig':market,'market_overround':overround-1,
            'quote':q,'unbuffered_fair_decimal_odds_by_route':{k:[1/v for v in p] for k,p in route_p.items()},
            'point_estimate_expected_net_return_by_route':{k:[v*o-1 for v,o in zip(p,odds,strict=True)] for k,p in route_p.items()},
            'market_favorite_index':favorite,'market_favorite_break_even_probability':inv[favorite],
            'market_favorite_fail_probability_by_route':{k:1-p[favorite] for k,p in route_p.items()},
            'fact_packet_record':context[f['match_id']],'llm_judgment':a,
            'distributions':{k:s[k] for k in ROUTES},'actual_outcome':None})
    for route,recs in cohorts.items():
        write(OUT/f'前瞻记录_{route}.json',{'pilot_id':f'serie-a-today-20261028-{route}','assembled_at':assembled.isoformat(),
            'records':recs,'results':{},'observation_unit':'five unique fixtures; repeated LLM/market across variants are not extra matches'})
        write(OUT/f'初始评分_{route}.json',score_multi(recs,{}))
    combined={'assembled_at':assembled.isoformat(),'class_order':['home','draw','away'],'unique_fixtures':5,
        'rows':rows,'primary_for_record':'goal_simple','combined_probabilities':None,'results':{},
        'primary_choice_reason':'Existing same-day goal source; rich lead3-to-lead0 transfer unvalidated',
        'new_statistical_fits_in_this_report':0,'existing_statistics_base_fits':2,'new_tree_fits':0,
        'parameter_searches':0,'new_llm_batches_in_this_report':0,'independent_llm_batches_reused':1,
        'parlay_recommendations':[]}
    write(OUT/'多线概率与价值对照.json',combined)
    lines=['# 今晚意甲预测','',
        '本批是意大利当地10月28日晚五场，北京时间10月29日01:30／03:45开球。[Lega Serie A官方赛程](https://www.legaseriea.it/serie-a/news/quando-si-gioca-anticipi-e-posticipi-fino-alla-12a-giornata)。',
        '',f'统计生成UTC：{stats["generated_at"]}；独立LLM生成：{generated}；市场快照：{quotes["observed_at"]}。报告组装：{assembled.isoformat()}，均早于开球。',
        '', '所有三概率按主胜／平局／客胜排列。进球主线、xG、121项研究、独立LLM与市场分别保留，没有平均、投票或采用新条件方向研究。',
        '', '|北京时间|比赛|进球主线%|独立xG%|121项研究%|独立LLM%|单源市场去水%|', '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        f=r['fixture'];p=r['probabilities_by_route']
        lines.append(f'|{time(f["kickoff"]).astimezone(BJ):%m-%d %H:%M}|{title(f)}|'+ '|'.join(pct(p[k]) for k in (*ROUTES,'llm_independent'))+'|'+pct(r['market_de_vig'])+'|')
    lines+=['','帕尔马—乌迪内斯方向存在分歧：进球主线与121项研究略偏乌迪内斯，独立xG和LLM略偏帕尔马，各自最高概率均不足40%，没有可靠稳胆。蒙扎—那不勒斯多数路线偏客，但LLM客胜仅49%，进球主线59.3%；保留这一幅度差异，不用共同方向提高置信度。']
    lines+=['','## 比分与总进球','', '以下全部来自进球主线，不从LLM三概率倒造比分。具体比分为概率最高的三项，仍只占分布的一部分。',
        '', '|比赛|预期进球主／客|三个比分及概率|0球%|1球%|2球%|3球%|4+球%|大2.5%|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        d=r['distributions']['goal_simple'];t=d['summary']['total_goals']
        scores='、'.join(f'{s["score"]}({100*s["probability"]:.1f}%)' for s in d['top_scores'][:3])
        lines.append(f'|{title(r["fixture"])}|{d["rates"][0]:.2f}／{d["rates"][1]:.2f}|{scores}|'+ '|'.join(f'{100*t[k]:.1f}' for k in ('0','1','2','3','4+'))+f'|{100*d["over_2_5"]:.1f}|')
    lines+=['','## 报价与公平赔率','',
        '报价来自单一DraftKings／ESPN快照。open和close是源字段；此时比赛未开始，close只称本次返回的latest，不能称最终收盘、实时可成交价或市场共识。源没有逐项更新时间和真实开盘时间，变化原因未知。',
        '', '|比赛|open主／平／客|latest主／平／客|去水前超额概率|','|---|---:|---:|---:|']
    for r in rows:lines.append(f'|{title(r["fixture"])}|{price(r["quote"]["opening_odds"])}|{price(r["quote"]["decimal_odds"])}|{100*r["market_overround"]:.2f}%|')
    lines+=['','各路线公平赔率分别为1/p，以下主／平／客全部列出。它们是不含缓冲的点估计盈亏平衡价，路线间差异不是置信区间；表面正差价尚未证明能盈利。',
        '', '|比赛|进球主线公平价|独立xG公平价|121项研究公平价|独立LLM公平价|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+title(r['fixture'])+'|'+'|'.join(price(r['unbuffered_fair_decimal_odds_by_route'][k]) for k in (*ROUTES,'llm_independent'))+'|')
    lines+=['','## 逐场依据、反证与风险','']
    outcomes=('主胜','平局','客胜')
    for r in rows:
        f=r['fixture'];p=r['probabilities_by_route'];i=r['market_favorite_index'];a=r['llm_judgment'];context=r['fact_packet_record']
        main_index=max(range(3),key=p['goal_simple'].__getitem__)
        favorite_name=f['home'] if i==0 else f['away'] if i==2 else '平局'
        fair=r['unbuffered_fair_decimal_odds_by_route']['goal_simple'][i]
        current=r['quote']['decimal_odds'][i]
        lines +=[f'### {title(f)}','',f'进球主线首选{outcomes[main_index]}，概率{100*p["goal_simple"][main_index]:.1f}%。市场概率最高项为{outcomes[i]}；该项在进球主线下失败概率为{100*(1-p["goal_simple"][i]):.1f}%，不能用“双选覆盖”掩盖单选风险。',
            '',f'该项latest为{current:.2f}，进球主线公平价为{fair:.2f}；'+('报价低于公平价，点估计下没有价格优势。' if current<fair else '报价高于公平价，仅构成模型点值的表面差价，尚非验证过的优势。'),'']
        for side in ('home','away'):
            t=context[side];s=t['observed_statistics'];v=s['relevant_venue']
            lines.append(f'{ZH[t["team"]]}：本季{s["season"]["wins"]}胜{s["season"]["draws"]}平{s["season"]["losses"]}负；近5场进／失球{s["last5"]["scored"]:.0f}／{s["last5"]["missed"]:.0f}，xG／xGA {s["last5"]["xG"]:.2f}／{s["last5"]["xGA"]:.2f}；本场相关主客场{v["matches"]}场、进／失{v["scored"]:.0f}／{v["missed"]:.0f}。')
        reason=a['reasoning']
        lines+=['','独立LLM支持：'+text(reason['support']),'','独立LLM反证：'+text(reason['contradictions']),'','不确定性：'+text(reason['uncertainty']),
            '', '当前伤停、停赛、临场战术和轮换未经本批独立核实，保持未知；联赛间隔不能代替完整欧战／杯赛负荷。','']
    lines+=['## 记录与使用范围','',
        '本批不强行给出二串／三串：独立路线尚未经验校准，当前报价也不是已验证优势。相同方向不代表误差独立，串关相乘还需要额外独立性假设。',
        '', '进球主线和独立xG使用截止10月28日前资料，各固定拟合一次；83场本季比分由原Football-Data80场及真实ESPN昨日3场补充，与Understat精确连接。原始CSV与昨日预测未改。',
        '', '丰富列直接复用原121项树，训练6,967场、其中意甲1,493场；训练快照提前3日，当前提前0日，外推表现未验证。独立LLM仅读事实包，概率为主观未校准估计；未知模型身份保留unknown。',
        '', '本报告训练0次、LLM重采0次，三组前瞻记录各5场而非15场独立比赛；初始评分已结算0场，准确率为空。各线完整三概率、比分网格与原始来源保留用于以后同场复核。半全场尚无完整模型，不填造。',
        '', '[多线概率与价格JSON](G:/Codex/football-prediction-2026-10-22/outputs/今日意甲预测_20261028/多线概率与价值对照.json)；[进球主线前瞻记录](G:/Codex/football-prediction-2026-10-22/outputs/今日意甲预测_20261028/前瞻记录_goal_simple.json)。']
    (OUT/'今晚意甲预测.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT/'今晚意甲预测.md'),'fixtures':5,'settled':0,'new_report_fits':0,
        'market_favorite_main_fair_odds':[(r['fixture']['home'],r['fixture']['away'],r['market_favorite_index'],r['quote']['decimal_odds'][r['market_favorite_index']],r['unbuffered_fair_decimal_odds_by_route']['goal_simple'][r['market_favorite_index']]) for r in rows]},ensure_ascii=False))

if __name__=='__main__':main()

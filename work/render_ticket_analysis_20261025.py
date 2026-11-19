"""Save a readable comparison of actual ticket routes; no refitting or settlement."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/用户六串一分析_20261025'

def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    ticket = read('票面与概率计算.json')
    epl = read('英超统计线路.json')
    europe = read('欧洲统计线路_资料刷新.json')
    statistical = {r['id']:r['probabilities'] for r in epl}
    for fid, row in zip(('7022','7024','7026'), europe['fixtures']):
        statistical[fid] = list(row['probabilities']['one_x_two'].values())
    findings = [
        '双选覆盖约六成，水晶宫周四欧联客场及锋线／门将疑问仍有风险。',
        '统计与LLM分歧：旧球队攻防模型仍给曼联约五成主胜；LLM较重视当下人员和周中表现。暂不作胆。',
        '两条预测线均偏拜仁，但约六成多客胜与1.28要求的78.1%差距明显；缺阵与强主场对手增加风险。',
        '主胜是最大单类，但统计54.2%与LLM66.0%分歧大，不能把1.37理解为稳胜。',
        '倾米兰，统计客胜仅50.7%；三个联赛客场连平、周四欧联客场负荷支持防平。',
        'MLS未接统计模型。LLM偏LAFC但并不认为1.43有优势；同城德比、银河近期反弹及LAFC平局即可锁定资格是反证。',
    ]
    records = []
    for r, finding in zip(ticket['fixtures'], findings):
        p = statistical.get(r['id'])
        records.append({**r, 'statistical_probabilities':p,
                        'statistical_selected_probability':None if p is None else sum(p[k] for k in r['selected']),
                        'joint_assessment':finding})
    now = datetime.now(timezone.utc).isoformat()
    evidence = [
        dict(fixture='7013',url='https://www.premierleague.com/en/news/4744431/predicted-line-ups-for-premier-league-teams-in-matchweek-8',
             claim='赛前伤停描述：Strand Larsen周中受伤、Henderson可用性未定；纽卡右路也存在人员问题。使用伤停叙述，不使用预测或确认首发。'),
        dict(fixture='7014',url='https://www.manutd.com/en/news/how-to-watch-and-follow-united-v-bournemouth-25-oct-26',
             claim='官方比赛预告确认赛程。英超官方赛前页同时记载曼联人员疑问及伯恩茅斯周四欧联负荷。'),
        dict(fixture='7022',url='https://fcbayern.com/en/news/matchpreviews/2026/10/preview-freiburg-vs.-bayern-bundesliga-md7',
             claim='拜仁预告确认Musiala、Díaz、Kim、Neuer缺阵及Kimmich休假；前三个联赛客场一胜两平，对手本季各项主场五连胜。'),
        dict(fixture='7024',url='https://www.levanteud.com/noticias/convocatoria-or-real-sociedad-levante-ud-2627',
             claim='莱万特赛前公告确认Bardeli、Dela缺席，比赛当地18:30。'),
        dict(fixture='7024',url='https://www.realsociedad.eus/bundles/equipo/partidos/real-sociedad',
             claim='皇家社会赛程与近期联赛信息；其最近三场联赛进攻旺盛但均失两球。'),
        dict(fixture='7026',url='https://www.acmilan.com/en/news/articles/serie_a/2026-10-25/udinese-vs-ac-milan-match-preview',
             claim='负责该场的研究上下文读取到赛前预告；root两次直接正文抓取失败。最终不依赖未在root正文重现的具体伤停名单；客场三连平另由新下载CSV支持。'),
        dict(fixture='7029',url='https://www.lafc.com/news/preview-lafc-vs-la-galaxy-2026-mls-season-october-25',
             claim='常规赛末段：LAFC第六、银河第八，LAFC打平也可锁定季后赛；临时教练三战一胜两平。'),
        dict(fixture='7029',url='https://www.lagalaxy.com/news/match-preview-presented-by-digalert-la-galaxy-at-lafc-october-25-2026',
             claim='银河赛前预告核实当地10/25 18:00 PT及近八场14分的近期表现。'),
        dict(fixture='all-statistical',url='https://www.football-data.co.uk/downloadm.php',
             claim='既有四季真实进球历史与本轮新下载2026/27 D1/SP1/I1；下载记录及目标队覆盖保存在相邻JSON。'),
    ]
    combined = dict(generated_at=now, records=records, evidence=evidence,
        route_status={'statistical':'actual existing fixed .01/180 goal model computations; incomplete latest other-team coverage; not a calibrated live deployment',
                      'llm':'same-platform subjective forecasts from sourced context, not fully blind or calibrated',
                      'market':'user screenshot proportional de-vig only, not independently verified live price',
                      'joint':'qualitative consensus/disagreement; no new fitted fusion or complete six-leg statistical probability'},
        arithmetic=ticket, actual_fit_calls_this_task=6, stale_snapshot_fit_calls=3,
        refreshed_fit_calls=3, saved_epl_replay_fit_calls=0, parameter_searches=0,
        target_results_reviewed=False, target_lineups_used=False)
    (OUT/'多路线六场记录.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 用户六串一：统计、LLM与票面市场比较','',
        f'整理时间：{now}。全部六场尚未开球时计算；只分析票面，不结算旧预测。', '',
        '**结论：不支持原样作为六串一主推。多个选项只有约五至六成覆盖率，热门单胜的价格也未显示稳健价值。**','',
        '下表是“图中所选结果合计通过率”，不是每场最高单项，也不是已验证真实胜率。统计使用现有带收缩的球队攻防泊松模型；LLM概率是研究者主观判断。', '',
        '| 比赛 | 图中所选 | 统计 | LLM | 票面比例去水 | 联合判断 |',
        '|---|---|---:|---:|---:|---|']
    labels=['胜＋平','平＋负','客胜','主胜','客胜','主胜']
    for r,label in zip(records,labels):
        p=r['statistical_selected_probability']
        lines.append(f"| {r['name']} | {label} | {'未覆盖' if p is None else f'{100*p:.1f}%'} | {100*r['llm_selected_probability']:.1f}% | {100*r['market_selected_probability']:.1f}% | {r['joint_assessment']} |")
    lines += ['', '## 价格与串关', '',
        '| 单选 | 票面赔率 | 盈亏平衡所需胜率 | 统计 | LLM |',
        '|---|---:|---:|---:|---:|']
    for r in records[2:]:
        k=r['selected'][0]; p=r['statistical_selected_probability']
        lines.append(f"| {r['name']} | {r['prices'][k]:.2f} | {100/r['prices'][k]:.1f}% | {'未覆盖' if p is None else f'{100*p:.1f}%'} | {100*r['llm_selected_probability']:.1f}% |")
    lines += ['',
        f"按票面去水概率且近似独立，六腿全部覆盖约{100*ticket['market_all_selected_independence']:.2f}%；按本次LLM点值同样计算约{100*ticket['llm_all_selected_independence']:.2f}%。数值相近并不证明LLM已校准。MLS无统计模型，所以不提供完整六场统计概率。", '',
        '前两场均双选，共4注六串一。若每注2元、1倍则成本8元；选项全部覆盖时仅中其中一注，按乘积赔率返还约55.72—85.32元，实际奖金以规则及出票为准。', '',
        f"在同一票面去水与独立假设下，期望毛返还／成本约{100*ticket['market_expected_gross_payout_ratio']:.2f}%。它是这组报价的数学描述，不是已证实的真实返还率。", '',
        '拜仁1.28、皇家社会1.37、LAFC1.43分别需要78.1%、73.0%、69.9%的真实获胜概率才保本；当前各路线不支持把它们当作已识别的价值选择。倾向谁赢与该价格是否值得购买是两个判断。', '',
        '## 实际执行与限制', '',
        '- 英超：回放10月24日保存的1591场拟合，其中现季71场；四支目标球队各7场联赛已知，最后比赛为10月17—18日。无新拟合。',
        '- 德西意：先检查旧快照得到3次探索拟合，发现缺轮次；保留旧结果后，下载真实当前季CSV，再按完全相同参数做3次资料刷新拟合。新报告训练1276／1607／1587场；目标队各6／9／7场联赛补齐。没有参数搜索。',
        '- Football-Data文件更新时间为10月20—21日，尚不含10月23—24日部分其他队当轮赛果。目标队最近联赛齐全不等于所有对手信息实时齐全。',
        '- 统计模型未量化本次伤停、欧战旅途或比赛投入；这些仅进入LLM文字分析。LLM与统计并非两个独立数据世界。',
        '- 本轮上下文已含票面赔率，LLM不能声称完全盲评。相同平台的不同分析也不是多个独立供应商模型。',
        '- 联合线路本次标记共识与分歧，未训练新权重；不因多线一致就声称更准。',
        '- 票面21:01不构成已核实的博彩公司更新时间；未取得真实开盘到即时变化，不能推断资金流向。',
        '- 未使用目标确认首发，未查询目标实时比分或结果，未改模型参数或启动GAP研究。', '',
        '## 来源', '']
    lines += [f"- [{r['fixture']}：来源]({r['url']})：{r['claim']}" for r in evidence]
    lines += ['', '[完整三路线逐场概率](多路线六场记录.json) · [欧洲刷新统计](欧洲统计线路_资料刷新.json) · [票面数学](票面与概率计算.json)', '']
    (OUT/'这张单的多路线分析.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(OUT/'这张单的多路线分析.md'),'record_count':len(records),
        'selected_probabilities':[{'match':r['name'],'statistical':r['statistical_selected_probability'],
            'llm':r['llm_selected_probability'],'market':r['market_selected_probability']} for r in records]},ensure_ascii=False))

if __name__=='__main__':
    main()

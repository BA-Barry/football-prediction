"""Present completed forecasts without fitting, averaging, or settling anything."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import json
import math
import shutil

from run_current_rich_forecasts_20261026 import CHINESE

ROOT=REPOSITORY_ROOT
OUT=ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z'


def main():
    stats=json.loads((OUT/'完整预测与来源.json').read_text(encoding='utf-8'))
    source=ROOT/'work/current_llm_independent_20261026.json'
    llm=json.loads(source.read_text(encoding='utf-8'))
    llm_by={r['fixture_id']:r for r in llm['records']}
    if len(llm_by)!=10 or set(llm_by)!={r['fixture']['match_id'] for r in stats['records']}:
        raise ValueError('LLM and statistical fixture IDs differ')
    def pct(p):
        if len(p)!=3 or any(not math.isfinite(x) or not 0<=x<=1 for x in p) or abs(sum(p)-1)>1e-10:
            raise ValueError('Three valid class probabilities required')
        return ' / '.join(f'{100*x:.1f}' for x in p)
    def top(p): return max(range(3),key=p.__getitem__)
    rows=[]
    changes=[]
    for r in stats['records']:
        l=llm_by[r['fixture']['match_id']]
        p=r['rich_candidate']['probabilities']
        q=l['probabilities']
        row={'fixture':r['fixture'],'home_label':r['home_label'],'away_label':r['away_label'],
             'old_20261024':r['original_20261024'],'current_simple':r['refreshed_simple'],
             'current_rich_candidate':r['rich_candidate'],'current_llm':l,
             'same_highest_single_outcome':top(p)==top(q),
             'maximum_single_probability_gap_pp':100*max(abs(a-b) for a,b in zip(p,q)),
             'joint_probabilities':None,'current_market':None,'actual_outcome':None}
        rows.append(row)
        if not row['same_highest_single_outcome']:
            changes.append(r['fixture']['home']+'—'+r['fixture']['away'])
    shutil.copy2(source,OUT/'本次独立LLM预测.json')
    llm_md=source.with_suffix('.md')
    if llm_md.exists(): shutil.copy2(llm_md,OUT/'本次独立LLM说明.md')
    result={'assembled_at':datetime.now(timezone.utc).isoformat(),'statistical_as_of':stats['as_of'],
            'llm_generated_at':llm.get('generated_at'),'class_order':['home','draw','away'],
            'probabilities_averaged':False,'new_weights_learned':False,'results_settled':False,
            'independent_llm_source_metadata':{k:v for k,v in llm.items() if k!='records'},'records':rows}
    (OUT/'多线逐场对照.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 本轮实际预测与优化进展','',
        '预测记录覆盖北京时间10月31日20:30至11月3日04:00的下一轮10场。所有概率依次为主胜／平局／客胜，单位%；四舍五入可能使合计偏离100%。',
        '', '本轮实际完成：同日简单进球模型、丰富信息候选，以及未查看模型概率和盘口的LLM主观对照。原10月24日记录保留，没有生成新的平均融合概率。',
        '', '| 北京时间 | 比赛 | 今日简单模型 | 今日丰富候选 | 本次LLM |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        dt=datetime.fromisoformat(r['fixture']['kickoff']).astimezone(timezone(timedelta(hours=8)))
        name=CHINESE[r['home_label']]+'—'+CHINESE[r['away_label']]
        lines.append(f"| {dt:%m-%d %H:%M} | {name} | {pct(r['current_simple']['probabilities'])} | {pct(r['current_rich_candidate']['probabilities'])} | {pct(r['current_llm']['probabilities'])} |")
    lines+=['','## 模型实际补充了什么','',
        '简单线用截至10月25日的1600场英超真实比分拟合球队攻防。xG基专家使用2360场同源历史。两者都在10月26日拟合，仍沿用0.01收缩、180天时间权重。',
        '', '主要丰富候选复用已研究过的固定随机树：120棵、深度6、每叶至少60个观测；从6967场、五联赛2022/23至2025/26的原始赛前特征样本学习预测全场真实进球。输入121项，包括进球与xG基专家、最近5/15场真实进失球、xG/NPxG、PPDA、deep、主客场、趋势及缺失标记。比分到胜平负的转换仍用同一套独立泊松分布。',
        '', '本赛季80场比分和球队表现进入当前基专家及近期特征；丰富候选本身的带标签训练样本最新为2026年5月24日，没有把本赛季标签悄悄加入它的训练。它学习较通用的近期表现关系，但这种跨赛季适用性仍需要前瞻记录检验。',
        '', 'LLM读取当前原始事实后单独给概率，没有查看统计概率或盘口。它与模型共享事实来源，判断同向也不能被当成两份统计独立的证据。LLM概率未经过训练和历史校准。',
        '', '## 当前变化及使用方式','',
        '切尔西主胜45.3%→39.4%，伯恩茅斯主胜43.4%→49.3%，赫尔城主胜54.2%→47.7%；这些是同日候选与简单模型的差别，不能把变化本身称为提升。候选当前最大单项为曼城主胜55.4%，没有单项达到70%。',
        '', '模型与本次LLM最高单项不同的比赛：'+('；'.join(changes) if changes else '无')+'。三项原值均保留；不通过改称“不败”、手工平均或抬高置信度消除分歧。',
        '', '没有核实本次可执行即时赔率，因此本轮不作价值投注或串关建议。单项概率最高只代表排序，不能直接说明赔率有价值。',
        '', '## 尚缺什么','',
        '20队全赛事记录已归档133场正式完赛，但独立覆盖核查的宽日期接口返回400，当前20队28日窗口均保留未知。主要候选使用121项近期表现版本；161项全赛事版本仅作缺失输入诊断，未作为完整负荷模型交付。',
        '', '本轮没有新增伤停、替代球员能力、战术变化或赛事优先级的量化修正。已归档人员延续资料不能直接换算为实力加减分。',
        '', '历史丰富模型研究主要在提前3天发布，本次实际提前5—7个日历日，且是10月26日首次初始化拟合。这两点尚未建立同等效果，不能将历史成绩直接搬成本批准确率。',
        '', '## 下一步与本轮停止点','',
        '本轮停止增加算法和参数搜索，保存已生成概率。后续重点对照这批真实赛前记录的三项概率质量、单项命中和概率可靠性，再根据跨多场的系统偏差决定是否调整。10场只用于前瞻观察，不能单凭它证明总体提升；不会因为一两场输赢即时改权重。',
        '', '原始来源：Football-Data历史与当前CSV、官方FPL、ESPN已完成比分、Understat球队历史。最新10场由ESPN补充，全部80场再与Understat逐场同日、同队、同比分连接。来源路径与取得时间见完整预测与来源.json；旧记录普通逐值核对未改。',
        '', '全部新数据、模型、记录及脚本均保存在项目目录。本次4次固定拟合、0轮参数搜索；模型保存后复现一致，10场概率轴与概率和已核对，没有结算目标赛果。']
    (OUT/'本轮预测与优化结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT/'本轮预测与优化结论.md'),'disagreement_matches':changes,
        'llm_probs':[{'home':r['home_label'],'away':r['away_label'],'p':r['current_llm']['probabilities']} for r in rows]},ensure_ascii=False))


if __name__=='__main__': main()

"""Read exactly three isolated responses, report variation without ensembling."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import statistics

ROOT=REPOSITORY_ROOT
OUT=ROOT/'outputs/LLM近期信息敏感性_20261026T043949767182Z'
ORIGINAL=ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z/多线逐场对照.json'
CN={'Chelsea':'切尔西','Man United':'曼联','Bournemouth':'伯恩茅斯','Leeds':'利兹联',
    'Brentford':'布伦特福德',"Nott'm Forest":'诺丁汉森林','Coventry':'考文垂','Sunderland':'桑德兰',
    'Hull':'赫尔城','Ipswich':'伊普斯维奇','Man City':'曼城','Brighton':'布莱顿',
    'Tottenham':'热刺','Crystal Palace':'水晶宫','Aston Villa':'维拉','Fulham':'富勒姆',
    'Liverpool':'利物浦','Arsenal':'阿森纳','Newcastle':'纽卡斯尔','Everton':'埃弗顿'}


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def top(p): return max(range(3),key=p.__getitem__)


def gap(p,q): return max(abs(a-b) for a,b in zip(p,q))*100


def main():
    plan=read(OUT/'方法与停止点.json')
    prior=read(ORIGINAL)
    full=read(OUT/'inputs/packet_r1.json')
    if full!=read(OUT/'inputs/packet_r2.json'): raise ValueError('Repeated factual inputs differ')
    ids={r['match_id'] for r in full['fixtures']}
    responses={}
    receipts={}
    for key in ('r1','r2','r3'):
        path=OUT/f'responses/{key}.json'
        obj=read(path)
        if obj['information_cutoff']!=full['as_of'] or obj['class_order']!=['home','draw','away']:
            raise ValueError('Cutoff or axes differ')
        if datetime.fromisoformat(obj['generated_at'].replace('Z','+00:00')).tzinfo is None:
            raise ValueError('Response requires real offset timestamp')
        if obj['input_path'].replace('\\','/')!=str(OUT/f'inputs/packet_{key}.json').replace('\\','/'):
            raise ValueError('Response cites another packet')
        for forbidden in ('statistical_probabilities','market','other_llm'):
            if obj['exposure'][forbidden] is not False: raise ValueError('Response exposure mismatch')
        rows={r['fixture_id']:r for r in obj['records']}
        if len(rows)!=10 or len(obj['records'])!=10 or set(rows)!=ids:
            raise ValueError('Response IDs differ')
        for r in rows.values():
            p=r['probabilities']
            if (len(p)!=3 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x)
                or not 0<=x<=1 for x in p) or abs(sum(p)-1)>1e-10):
                raise ValueError('Invalid three-way response')
            if any(abs(100*x-round(100*x))>1e-8 for x in p):
                raise ValueError('Requested integer-percent subjective output violated')
        responses[key]=rows
        receipts[key]={k:v for k,v in obj.items() if k!='records'}
    details=[]
    for fixture in full['fixtures']:
        mid=fixture['match_id']
        a,b,c=[responses[k][mid]['probabilities'] for k in ('r1','r2','r3')]
        stable=top(a)==top(b)
        details.append({'fixture':fixture,'full_response_1':a,'full_response_2':b,'recent_two_withheld_response':c,
            'same_information_repeat_gap_pp':gap(a,b),'withheld_vs_full_1_gap_pp':gap(a,c),
            'withheld_vs_full_2_gap_pp':gap(b,c),'full_repeat_same_top_class':stable,
            'top_changes_after_withholding_when_full_agreed':stable and top(c)!=top(a),
            'notes':'These are within-probe responses, not replacements of published forecasts.'})
    summary={}
    for key in ('same_information_repeat_gap_pp','withheld_vs_full_1_gap_pp','withheld_vs_full_2_gap_pp'):
        values=[r[key] for r in details]
        summary[key]={'mean':statistics.mean(values),'median':statistics.median(values),'max':max(values)}
    summary.update(full_repeat_top_disagreements=sum(not r['full_repeat_same_top_class'] for r in details),
        top_changes_after_withholding_among_repeat_agreed=sum(r['top_changes_after_withholding_when_full_agreed'] for r in details),
        full_repeat_agreed_matches=sum(r['full_repeat_same_top_class'] for r in details))
    if read(ORIGINAL)!=prior: raise ValueError('Published comparison changed')
    result={'assembled_at':datetime.now(timezone.utc).isoformat(),'method_file':'方法与停止点.json',
        'summary':summary,'records':details,'response_metadata':receipts,
        'new_llm_responses':3,'resampling_or_selection':False,'new_model_fits':0,'parameter_searches':0,
        'probabilities_averaged_or_ensembled':False,'target_outcomes_scored':False,'original_forecasts_unchanged':True,
        'limits':plan['limits']}
    write(OUT/'三次响应敏感性对照.json',result)
    lines=['# LLM对最近两场信息的敏感性：一次有限对照','',
        '使用同一批10场未开赛目标、同一信息截点。两次给予完全相同的完整资料，另一次隐藏每队最近两场完整比赛后重新计算所有汇总。三次任务彼此不提供对方回答，也不提供统计模型或市场概率。',
        '', '本次固定3次响应，不重试挑答案、不训练、不调参、不结算赛果，不替换原LLM或统计预测，也没有平均融合。',
        '', '| 对照 | 每场最大单项变化的平均值 | 最大值 |','|---|---:|---:|']
    labels={'same_information_repeat_gap_pp':'相同完整资料重复一次',
            'withheld_vs_full_1_gap_pp':'隐藏最近两场 对 完整响应1',
            'withheld_vs_full_2_gap_pp':'隐藏最近两场 对 完整响应2'}
    for k,label in labels.items():
        v=summary[k]
        lines.append(f"| {label} | {v['mean']:.2f}个百分点 | {v['max']:.2f}个百分点 |")
    lines+=['','每场先取主胜、平局、客胜三项中最大的绝对差，再对10场求平均。它不是准确率或置信区间，也不是LLM总方差。',
        '',f"相同资料两次最高单项不同：{summary['full_repeat_top_disagreements']}/10。两次原本同向的{summary['full_repeat_agreed_matches']}场中，隐藏最近两场后最高单项改变：{summary['top_changes_after_withholding_among_repeat_agreed']}场。",
        '', '| 比赛 | 完整资料响应1 主／平／客 | 完整资料响应2 | 隐藏最近两场 | 同资料差／隐藏差1／隐藏差2（百分点） |',
        '|---|---:|---:|---:|---:|']
    for r in details:
        f=r['fixture'];title=CN[f['home_label']]+'—'+CN[f['away_label']]
        def pct(p): return '／'.join(f'{100*x:.0f}' for x in p)
        gaps='／'.join(f"{r[k]:.0f}" for k in ('same_information_repeat_gap_pp','withheld_vs_full_1_gap_pp','withheld_vs_full_2_gap_pp'))
        lines.append(f"| {title} | {pct(r['full_response_1'])} | {pct(r['full_response_2'])} | {pct(r['recent_two_withheld_response'])} | {gaps} |")
    lines+=['','## 能说明什么','',
        '同一资料下的差异是本次可见的重复判断波动。隐藏信息与完整资料的差异同时含资料效应、推理方式差异及生成波动；不能简单相减后当成纯近期效应，更不能由此求出“最近两场占多少权重”。',
        '', '这份资料包限制为数值比赛历史；它与原先当前LLM使用的完整提示不同，因此本次只评价这套受控提示，不能回写为原LLM早先的真实概率。球队名称仍在资料中，虽已要求不用名声与外部记忆，不能技术上排除内部先验。',
        '', '隐藏完整比赛会同时移除比分和xG/NPxG、减少本季样本，并使更早比赛补入最近窗口；不是只测试输赢，也不是球队真实实力的因果实验。实际每队移除2场、合计20场独立比赛，完整资料每队本季8场、隐藏后6场。',
        '', '三次代理沿用父任务配置，没有指定不同模型，但实际模型ID与采样设置无法独立核实。相同资料仅重复一次，不能估计稳定分布、显著性或安全阈值。当前目标没有赛果评分，因此不能判断哪次更准。',
        '', '此前统计候选隐藏最近两场的平均最大单项变化约1.48个百分点，本次LLM约5.6—5.8个百分点；两项试验的资料呈现、拟合通道和提示不同，只能分别说明局部敏感性，不能作为准确率排名或严格算法对比。',
        '', '## 本轮执行决定','',
        '现有统计预测和原独立LLM记录保持不变。LLM概率继续单独列示，不因同向而上调置信度，也不把一次重问后的答案当成更优版本。后续复核时按原发布时间分别评分，保留本次三次响应作为敏感性诊断。',
        '', '本轮到3次响应为止，停止追加抽样和提示词搜索。全部输入、响应、方法及核验位于项目目录。']
    report=OUT/'LLM近期信息影响说明.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(report),'summary':summary},ensure_ascii=False))


if __name__=='__main__': main()

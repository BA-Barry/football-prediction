"""Read-only diagnostics and source-only LLM examples; no new model fits."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from run_shot_increment_20261025 import score_details

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
OUT = ROOT/'outputs/历史情境共同接入_20261028'
DATA = ROOT/'data/historical_context_20261028'
BASE = ROOT/'outputs/协作预测与大样本方案_20261028/扩样训练'


def read(path):
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as f:
            return json.load(f)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def top(row):
    return max(range(3),key=lambda i:row['probabilities'][i])


def main():
    result=read(OUT/'共同接入训练结果.json')
    old=read(BASE/'扩样_20250801_2025-26预测.json')
    new=read(OUT/'共同情境_2025-26逐场预测.json')
    transitions=Counter()
    deltas=[]
    for a,b in zip(old,new,strict=True):
        if (a['match_id'],a['actual_score'])!=(b['match_id'],b['actual_score']):
            raise ValueError('Paired research predictions must have identical targets')
        x,y,z=top(a),top(b),a['actual_outcome']
        category='both_correct' if x==z and y==z else 'corrected' if y==z else 'damaged' if x==z else 'both_wrong'
        transitions[category]+=1
        transitions['top_changed']+=x!=y
        deltas.append(max(abs(p-q) for p,q in zip(a['probabilities'],b['probabilities'],strict=True)))
    by_league={}
    for comp in sorted({r['competition'] for r in new}):
        a=[r for r in old if r['competition']==comp]
        b=[r for r in new if r['competition']==comp]
        by_league[comp]={'n':len(a),'old':score_details(a)[0]['one_x_two'],
            'joint':score_details(b)[0]['one_x_two']}
    diagnostics={'generated_at':datetime.now(timezone.utc).isoformat(),'n':len(new),
        'top1_transitions':dict(transitions),'mean_max_single_outcome_probability_change':sum(deltas)/len(deltas),
        'largest_single_outcome_probability_change':max(deltas),
        'top1_prediction_counts':{'old':dict(Counter(top(r) for r in old)),
            'joint':dict(Counter(top(r) for r in new))},'by_league':by_league,
        'interpretation':'Post-hoc description only, not forecast routing, causal attribution or a per-league weight search'}
    write(OUT/'共同接入变化说明.json',diagnostics)
    # Examples are selected by source availability/observed coaching context,
    # never by a target goal or whether a forecast was correct.
    original=read(BASE/'扩样合并特征.json.gz')['rows']
    coaches={r['match_id']:r for r in read(DATA/'coach_sources/coach_context_lead3.json.gz')['rows']}
    status={r['match_id']:r for r in read(DATA/'availability_sources/8753场历史可用性上下文.json.gz')['rows']}
    schedule={r['match_id']:r for r in read(DATA/'schedule_sources/8753场历史全赛事上下文.json.gz')['rows']}
    chosen=[]
    for comp in ('E0','I1','D1','SP1','F1'):
        candidates=[r for r in original if r['competition']==comp and r['period']=='2024-25']
        chosen.append(next((r for r in candidates if any(
            s['features']['observed_change_age_days'] is not None and s['features']['consecutive_same_name_games']<=3
            for s in coaches[r['match_id']]['coach_context'])),candidates[0]))
    examples=[]
    for row in chosen:
        ident=row['match_id'];c,a,s=coaches[ident],status[ident],schedule[ident]
        examples.append({k:row[k] for k in ('match_id','date','period','competition','home','away','forecast_before_date')}
            |{'statistical_rates_at_earlier_parameter_date':{'goal':row['goal_rates'],'xg':row['xg_rates'],
                'parameter_before':row['fit_before_date']},
                'coach_prior_observations':c['coach_context'],
                'archived_player_status':{'archive_observation':a['archive_observation'],
                    'teams':a['teams'],'missing_reason':a['missing_reason']},
                'past_schedule':{'home':s['home_context'],'away':s['away_context']}})
    write(OUT/'历史情境LLM资料理解样例.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Historical source interpretation examples only; not retrospective forecasts or evaluated LLM accuracy',
        'selection':'One 2024/25 fixture per league, prefer a recently observed coach-label change; no outcome selection',
        'target_labels_exported':False,'target_manager_exported':False,'llm_calls':0,
        'instructions':['Separate source fact, interpretation and unknown.',
            'Last known coach is not confirmed target coach; source label change is not appointment announcement.',
            'News and prior minute shares do not directly supply a lost-strength coefficient.',
            'These dates are historical. Do not produce a purported authentic historical prematch probability.'],
        'records':examples})
    a=result['old_expanded121']['one_x_two'];b=result['joint_context213']['one_x_two']
    lines=['# 历史情境共同接入：本轮结果与所学','',
        '2026-10-28。所有数据、模型、缓存和报告均在G盘。本轮到两次固定设置训练及一次独立核对结束；不追加调参或替换当前预测。','',
        '## 真正新增与复用的内容','',
        '|信息|实际覆盖|本轮处理|','|---|---|---|',
        '|原121项表现|8753场|逐值保留，继续学真实全场进球|',
        '|跨赛事历史安排|6967场双方范围覆盖；双方28日窗口6913场可用|复用既存资料，每侧40项；2021/22缺失保留未知|',
        '|历史球员状态|1502场英超、459份既存快照|复用14项状态计数及22项较早参与变化；其他四联赛和2021/22未知|',
        '|较早比赛中的教练观测|8753场、17506侧|新接作者公开归档，每队8项；自己及对手共16项|','',
        '合计每侧213项。教练包按明确球队ID与日期联结，8752场原日期精确匹配，1场归档日期错误经马竞官网核实显式修正；原下载保留。数值输入没有目标比赛教练、姓名编码、目标首发、目标分钟或手写换帅加成。','',
        '教练归档来自[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)，末日2026-07-06，不能当作2026/27当前教练资料。球员快照复用[Randdalf/fplcache](https://github.com/Randdalf/fplcache)。资料的历史首次发布时间和修订未重建；Git提交日期只是回溯版本代理。仓库软件许可不扩大底层数据权利。','',
        '## 固定共同学习，而非逐项试输赢','',
        '沿用原ExtraTrees进球学习器、树设置、180天衰减及独立Poisson比分转换。状态、参与、赛程和教练历史一起输入，让同一棵树的分裂有机会学习共同关系。没有给信息人工分配权重，也没有各做一轮单因素筛选后删方向。','',
        '一次模型仅训练2025-08-01之前的7027场；复用已有相同训练范围121项模型，在同一1726场2025/26比赛比较。第二次用2026-08-01之前全部8753场保存研究候选；它没有用于该批持出评分。整个主流程约84.90秒，新增基础拟合0、树拟合2、参数搜索0。','',
        '|指标|原扩样121项|共同情境213项|','|---|---:|---:|',
        f"|胜平负首选|{round(a['accuracy']*a['n'])}/{a['n']}（{a['accuracy']:.2%}）|{round(b['accuracy']*b['n'])}/{b['n']}（{b['accuracy']:.2%}）|",
        f"|胜平负LogLoss，越低越好|{a['log_loss']:.6f}|{b['log_loss']:.6f}|",
        f"|胜平负Brier，越低越好|{a['brier_sum']:.6f}|{b['brier_sum']:.6f}|",
        f"|比分LogLoss|{result['old_expanded121']['exact_score']['log_loss']:.6f}|{result['joint_context213']['exact_score']['log_loss']:.6f}|",
        f"|总球分组LogLoss|{result['old_expanded121']['total_goals_0_1_2_3_4plus']['log_loss']:.6f}|{result['joint_context213']['total_goals_0_1_2_3_4plus']['log_loss']:.6f}|",'',
        f"首选变化{transitions['top_changed']}场，纠正{transitions['corrected']}、改坏{transitions['damaged']}；平均每场最大的单项概率变化{sum(deltas)/len(deltas)*100:.3f}个百分点。概率评分微幅改善而首选减少3场，不能合称明显提准。按联赛的完整点值及变化见`共同接入变化说明.json`，不据事后分组挑选部署线路。",'',
        '这些年份已经参与研发，成绩只是已查看历史上的固定整合测量，不能称新盲测或当前命中率。两种模型共享相同目标、较早基础均值及原121项；增加信息会改变树的随机分裂机会，差值也不是各因素的因果效果。','',
        '## 本轮具体学到了什么','',
        '1. 数据能共同学习，并不需要人为把模型和LLM概率平均。但输入含义决定能学到什么：目前球员块主要是源状态和此前使用，教练块是较早观测连续性，尚不是个人能力损失或战术质量。','',
        '2. 8753是比赛数，不是8753场都有完整情境。只有1502场有FPL，2021/22没有本包全赛事资料；不能把其余场的缺失当零伤停或零负荷。历史五联赛当前球员信息仍是实质缺口。','',
        '3. 观测到过去教练改变，有助于标记历史可迁移性的不确定性；它无法确认比赛之间已官宣的新教练，也不能证明换帅收益。8项数值和一条源姓名标签不等于战术模拟。','',
        '4. 新字段并不自动带来大幅信息增量。FPL和赛程此前已经接通过，本轮价值是统一截止、保留未知并建立共同学习入口，加上教练历史；没有将资料整理说成全新信息发现。','',
        '5. 本轮仍只改变主客进球均值的输入，保留原比分分布。胜平负排序受整个输出结构影响，不能从这一次首选下降就判定情境无效；也不应据微小LogLoss改善宣称联合已经验证成功。','',
        '6. LLM作用仍是解释非结构化证据、提出反证并辨别重复信息；这次没有生成数千份伪历史LLM预测。另已导出五联赛各一份不含目标标签的资料理解样例，只供文本提取/流程研究，不充作历史前瞻。','',
        '## 后续顺序与本轮停止点','',
        '保留共同输入、两个研究模型、全部来源和失败/纠正记录；本轮不再更换衰减、树深、权重或按联赛救分。当前赛前统计、独立LLM、情境LLM及助手综合继续保留各自身份，五场原记录未变，未查询新赛果。','',
        '下一步先明确这些信息如何服务当前季状态与证据解释：有可靠历史字段的进入共同数据；新闻的严重性、持续期、人员替代与战术适应没有量化依据时，交给LLM说明作用路径和未知，不假装已训练因果效应。若要改变长期规律和短期状态的学习关系，或将文本解释学成胜率修正，应先讨论具体结构及所需监督资料。','',
        '核对见[独立核对](G:/Codex/football-prediction-2026-10-22/outputs/历史情境共同接入_20261028/独立核对.md)：原121项2118226单元保持，892373条教练引用严格更早；评分训练7027与目标1726的ID交集0；训练预处理仅由训练数据重算一致，保存模型全目标重放率误差0。']
    (OUT/'本轮共同接入与所学.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'diagnostics':diagnostics['top1_transitions'],
        'mean_max_probability_change':diagnostics['mean_max_single_outcome_probability_change'],
        'examples':len(examples),'fits':0,'report':str(OUT/'本轮共同接入与所学.md')},ensure_ascii=False))


if __name__=='__main__':
    main()

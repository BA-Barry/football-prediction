"""Describe all saved draw-branch errors by fixed prematch conditions.

No fit, forecast refresh, market/LLM inference, result query or decision rule.
Groups are descriptive, not selected deployment subsets.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT / 'outputs/历史比赛模式与平局分支_20261027'
OUT = ROOT / 'outputs/平局条件信息定位_20261027'
LABELS = ('主胜', '平局', '客胜')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare():
    records = read(SOURCE / '逐场概率与纠错.json')['records']
    with gzip.open(SOURCE / '新增历史模式输入与来源.json.gz', 'rt', encoding='utf-8') as stream:
        source = json.load(stream)['rows']
    index = {row['match_id']: row for row in source}
    if len(index) != len(source) or len({row['match_id'] for row in records}) != len(records):
        raise ValueError('Duplicate archived match')
    result = []
    for row in records:
        pattern = index[row['match_id']]
        if any(row[k] != pattern[k] for k in ('date', 'fit_before_date', 'forecast_before_date')):
            raise ValueError('Pattern and forecast timing differ')
        p = np.asarray(row['original_probabilities'])
        q = np.asarray(row['probabilities'])
        previous = np.asarray(row['previous_branch_probabilities'])
        actual = row['actual_outcome']
        original_top, new_top = int(p.argmax()), int(q.argmax())
        decisive = np.asarray((p[0], p[2])) / (p[0]+p[2])
        dominance = float(decisive.max())
        required = dominance/(1+dominance)
        d = float(q[1])
        if not np.allclose(q[[0, 2]]/(1-d), decisive, rtol=0, atol=1e-12):
            raise ValueError('Archived branch does not retain conditional H/A')
        if abs(d-required) > 1e-12 and ((new_top == 1) != (d > required)):
            raise ValueError('Draw top1 boundary differs from archived decision')
        corrected = original_top != actual and new_top == actual
        damaged = original_top == actual and new_top != actual
        changed = original_top != new_top
        history = pattern['pattern_history']['history']
        full15 = all(history[side]['windows']['15']['count'] == 15 for side in ('home', 'away'))
        total = math.fsum(row['source_original_rates'])
        result.append(row | {
            'conditional_HA_dominance': dominance, 'draw_top1_boundary': required,
            'draw_probability_minus_boundary': d-required,
            'original_top1': original_top, 'new_top1': new_top,
            'corrected': corrected, 'damaged': damaged, 'changed': changed,
            'change_type': ('corrected' if corrected else 'damaged' if damaged
                else 'both_wrong' if changed else 'unchanged'),
            'original_non_draw_direction_wrong': actual != 1 and original_top != actual,
            'original_total_mean': total,
            'strength_group': 'near_even_HA' if dominance <= .6 else 'moderate_HA' if dominance <= .75 else 'strong_HA',
            'total_group': 'below_2.5' if total < 2.5 else '2.5_to_3.5' if total <= 3.5 else 'above_3.5',
            'history_group': 'both_full15' if full15 else 'one_or_both_short15',
            'new_minus_original_log_loss': math.log(float(p[actual]/q[actual])),
            'new_minus_previous_log_loss': math.log(float(previous[actual]/q[actual])),
        })
    return result


def describe(rows):
    labels = np.asarray([r['actual_outcome'] for r in rows])
    draw = (labels == 1).astype(int)
    tracks = {}
    for name, key in [('original', 'original_probabilities'), ('previous', 'previous_branch_probabilities'),
                      ('new', 'probabilities')]:
        p = np.asarray([r[key] for r in rows])
        top = p.argmax(axis=1)
        tracks[name] = {'top1_correct': int((top == labels).sum()), 'accuracy': float((top == labels).mean()),
            'log_loss': float(-np.log(p[np.arange(len(rows)), labels]).mean()),
            'draw_mean_probability': float(p[:, 1].mean()),
            'draw_auc': float(roc_auc_score(draw, p[:, 1])) if len(set(draw)) == 2 else None,
            'draw_top1_n': int((top == 1).sum()),
            'draw_top1_correct': int(((top == 1) & (labels == 1)).sum())}
    return {'n': len(rows), 'actual_outcomes': np.bincount(labels, minlength=3).tolist(),
        'draw_frequency': float(draw.mean()), 'tracks': tracks,
        'changes_vs_original': dict(Counter(r['change_type'] for r in rows)),
        'original_wrong_non_draw_unfixable_by_draw_branch': sum(r['original_non_draw_direction_wrong'] for r in rows),
        'mean_required_draw_top1_probability': float(np.mean([r['draw_top1_boundary'] for r in rows])),
        'mean_new_draw_probability': float(np.mean([r['probabilities'][1] for r in rows])),
        'mean_new_minus_original_LL': float(np.mean([r['new_minus_original_log_loss'] for r in rows])),
        'mean_new_minus_previous_LL': float(np.mean([r['new_minus_previous_log_loss'] for r in rows]))}


def decompose(rows):
    """Separate draw-vs-not-draw and conditional H/A terms, not new forecasts."""
    selected = [r for r in rows if r['actual_outcome'] != 1]
    old_top, new_top, labels = [], [], []
    losses = {name: {'draw_term': [], 'direction_term': [], 'full': []}
        for name in ('original', 'unused_classifier')}
    for row in rows:
        actual = row['actual_outcome']
        for name, key in [('original', 'original_probabilities'),
                          ('unused_classifier', 'classifier_probabilities_reference_only')]:
            p = row[key]
            draw_term = -math.log(p[1] if actual == 1 else p[0]+p[2])
            direction_term = 0. if actual == 1 else -math.log(p[actual]/(p[0]+p[2]))
            full = -math.log(p[actual])
            if abs(draw_term+direction_term-full) > 1e-12:
                raise ValueError('Outcome-loss factorization failed')
            for key2, value in [('draw_term', draw_term), ('direction_term', direction_term), ('full', full)]:
                losses[name][key2].append(value)
        if actual != 1:
            old_top.append(0 if row['original_probabilities'][0] >= row['original_probabilities'][2] else 2)
            new_top.append(0 if row['classifier_probabilities_reference_only'][0] >= row['classifier_probabilities_reference_only'][2] else 2)
            labels.append(actual)
    return {'all_n': len(rows), 'observed_non_draw_n': len(selected),
        'scope': 'conditional direction diagnosis on known historical non-draw outcomes; not deployable selection or future accuracy',
        'original_direction_correct': sum(a == y for a, y in zip(old_top, labels)),
        'classifier_direction_correct': sum(a == y for a, y in zip(new_top, labels)),
        'classifier_corrected_direction': sum(a != y and b == y for a, b, y in zip(old_top, new_top, labels)),
        'classifier_damaged_direction': sum(a == y and b != y for a, b, y in zip(old_top, new_top, labels)),
        'mean_terms_over_all_fixtures': {name: {key: float(np.mean(value)) for key, value in pieces.items()}
            for name, pieces in losses.items()},
        'mean_conditional_direction_LL_on_observed_non_draws': {
            name: math.fsum(pieces['direction_term'])/len(selected) for name, pieces in losses.items()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write('本轮定位范围.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'input': str(SOURCE), 'training_fits': 0, 'parameter_searches': 0, 'new_result_queries': 0,
        'scope': 'retrospective description of all existing rows; not feature selection or validation',
        'strength_groups': 'conditional max(H,A)/(H+A): <=.6, (.6,.75], >.75',
        'total_groups': 'original total mean <2.5, [2.5,3.5], >3.5',
        'history_groups': 'both last15 full, otherwise shorter; no target omission',
        'stop': 'one complete saved-record analysis, source coverage inspection; no subgroup-specific rule/weights'})
    rows = prepare()
    summary = {}
    for period in ('2024-25', '2025-26'):
        local = [r for r in rows if r['period'] == period]
        grouped = {}
        for field in ('strength_group', 'total_group', 'history_group'):
            grouped[field] = {key: describe([r for r in local if r[field] == key])
                for key in sorted({r[field] for r in local})}
            if sum(r['n'] for r in grouped[field].values()) != len(local):
                raise ValueError('Descriptive groups omitted or duplicated targets')
        summary[period] = {'all': describe(local), 'by_condition': grouped,
                           'direction_components': decompose(local)}
    write('全部历史条件定位.json', {'by_period': summary, 'records': rows})
    changed = [r for r in rows if r['changed']]
    fields = ['match_id', 'period', 'date', 'competition', 'home', 'away', 'actual_score',
        'actual_outcome', 'change_type', 'original_home', 'original_draw', 'original_away',
        'new_home', 'new_draw', 'new_away', 'conditional_HA_dominance', 'draw_top1_boundary',
        'draw_probability_minus_boundary', 'original_total_mean', 'history_group']
    with (OUT / '全部改首选比赛.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in changed:
            value = {key: row[key] for key in fields if key in row}
            for prefix, key in [('original', 'original_probabilities'), ('new', 'probabilities')]:
                value.update(dict(zip((prefix+'_home', prefix+'_draw', prefix+'_away'), row[key])))
            writer.writerow(value)
    lines = ['# 平局分支的条件定位', '',
        '本轮只使用已保存预测及历史结果，未训练、调参、生成新预测或查询当前赛果。'
        '两季全部3,452场保留；以下切片用于解释错误，不能当作可上线的筛选规则。', '',
        '设原模型在非平条件下较强一方的赢球比例为r=max(pH,pA)/(pH+pA)。'
        '只替换平局分量d时，要让平局成为唯一最高概率结果，必须d>r/(1+r)。'
        '这是一条概率排序的代数边界，不是新投注阈值，也不说明应该人为抬高d。'
        '如果真实条件平局概率低于主胜或客胜，按最高概率单选时本就应选择后者；'
        '没有大量平局首选本身不能证明预测失败，强行增加平局首选也不保证提高命中率。', '',
        '| 赛季 | 原模型错判 | 其中实际平局 | 其中主客方向反选 | 新分支纠正 | 新分支改坏 |',
        '|---|---:|---:|---:|---:|---:|']
    for period, value in summary.items():
        a = value['all']
        lines.append(f"| {period} | {a['n']-a['tracks']['original']['top1_correct']} | "
            f"{a['actual_outcomes'][1]} | {a['original_wrong_non_draw_unfixable_by_draw_branch']} | "
            f"{a['changes_vs_original'].get('corrected',0)} | {a['changes_vs_original'].get('damaged',0)} |")
    lines += ['', '原丰富模型本来没有平局首选，因此上述实际平局全部是其单选错误。'
        '保持主客胜比例的平局分支能改变平／非平选择，却无法把这批主客方向反选直接改对。'
        '这限制的是本分支的职责，不代表整个模型或其他算法存在同样上限。', '',
        '| 赛季 | 原主客相对强弱组 | 场次 | 实际平局率 | 新平均平局概率 | 新平局首选：中/选 | 新减原LL |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels = {'near_even_HA': 'r≤0.60', 'moderate_HA': '0.60<r≤0.75', 'strong_HA': 'r>0.75'}
    for period, value in summary.items():
        for group, a in value['by_condition']['strength_group'].items():
            n = a['tracks']['new']
            lines.append(f"| {period} | {labels[group]} | {a['n']} | {100*a['draw_frequency']:.2f}% | "
                f"{100*n['draw_mean_probability']:.2f}% | {n['draw_top1_correct']}/{n['draw_top1_n']} | "
                f"{a['mean_new_minus_original_LL']:+.6f} |")
    lines += ['', '分组使用原赛前概率，未利用赛果分组或按子组选择模型。'
        '相对强弱是在排除平局条件下定义的r，不能简单等同于两队现实实力。'
        '全部总球与历史覆盖切片同时保存在JSON，不据后季结果定制权重。', '',
        '## 没有采用的主客分量里学到了什么', '',
        '胜平负对数损失可拆为“平／非平损失＋仅在非平结果上的主／客条件损失”。'
        '本轮还读取了保存但未采用的分类树主客分量，只用于定位互补，不生成新预测。', '',
        '| 赛季 | 历史实际非平场数 | 原主客方向正确 | 分类树主客方向正确 | 独有纠正 | 改坏 | 原条件LL | 分类树条件LL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for period, value in summary.items():
        a = value['direction_components']
        ll = a['mean_conditional_direction_LL_on_observed_non_draws']
        lines.append(f"| {period} | {a['observed_non_draw_n']} | {a['original_direction_correct']} | "
            f"{a['classifier_direction_correct']} | {a['classifier_corrected_direction']} | "
            f"{a['classifier_damaged_direction']} | {ll['original']:.6f} | {ll['unused_classifier']:.6f} |")
    lines += ['', '分类树两季都有原模型未判对的主客方向，不能说它没有可学之处；'
        '但它也改坏原来正确的判断，条件概率损失没有改善。此表事后限定实际非平比赛，'
        '不能把条件正确率当成完整胜平负准确率，也不能在未来预先知道该用哪一条。'
        '当前保留双方输出，未改变已批准的主客分工。', '',
        '## 所有改首选的比赛', '',
        'H/D/A依次为主胜／平局／客胜，分别列出，不以不败相加替代单选。', '',
        '| 赛季／日期 | 对阵 | 比分 | 原H/D/A | 新H/D/A | 变化 |',
        '|---|---|---|---|---|---|']
    names = {'corrected': '纠正', 'damaged': '改坏', 'both_wrong': '前后均错'}
    for r in changed:
        p = '/'.join(f'{100*v:.1f}%' for v in r['original_probabilities'])
        q = '/'.join(f'{100*v:.1f}%' for v in r['probabilities'])
        home, away = (r[key].split(':', 2)[-1] for key in ('home', 'away'))
        lines.append(f"| {r['period']}／{r['date']} | {home}—{away} | "
            f"{r['actual_score'][0]}:{r['actual_score'][1]} | {p} | {q} | {names[r['change_type']]} |")
    lines += ['', '这批比赛不足以单独识别人员、战术或状态的因果影响。'
        '本轮不从这些已发生结果反推一条过滤规则，也不重新生成当时的LLM意见。', '',
        '## 已有信息与本次补充', '',
        '旧项目已有最近5场“实际净xG−该旧场赛前预计净xG”，'
        '不能把对手校正重新包装成首次加入。它尚不等于按领先／落后、红牌人数差调整历史机会。'
        '整场xG均值和终场比分无法还原这些状态；本轮已完成以下实际资料测量，'
        '未知部分保持未知，未训练新候选。', '',
        f'[全部条件结果]({OUT.as_posix()}/全部历史条件定位.json) · '
        f'[全部改首选比赛]({OUT.as_posix()}/全部改首选比赛.csv)', '',
    ]
    shot_path = OUT / '射门比分状态覆盖.json'
    minute_path = OUT / 'EPL分钟资料精确联结.json'
    if shot_path.exists():
        c = read(shot_path)['counts']
        lines += ['## 已有逐射门资料：实际可恢复多少', '',
            f"已对英超2023/24、2024/25共{c['matches']:,}场、{c['shots']:,}条射门实跑研究解析器。"
            f"其中{c['early_unique_own_state']:,}条可唯一判断本方射门前领先／持平／落后，"
            f"{c['early_ambiguous_own_state']:,}条早段状态仍有歧义；"
            f"{c['minute_ge45_unknown']:,}条因缺可靠半场和事件时钟保留未知。", '',
            f"只在来源采用全场累计分钟的前提下解释minute<45；不会按ID或源数组推断先后。"
            f"{c['own_goal_events']}次乌龙按相反一方记账后，{c['matches']}/{c['matches']}场进球总数与独立终场一致。"
            '源h_goals/a_goals在本批全部为终场字段，完全不用于射前状态。'
            '新增组件6项针对性测试通过；这些结果只说明覆盖与接入正确性，不是预测提准。', '',
            '已保留每条可能比分集合和本方状态；比分不唯一但状态唯一时明确区别。'
            '原始单次射门xG不直接求和替代源NPxG，不把未知填0:0，也不把早段覆盖外推为全场状态。', '',
            f'[逐射门测量说明]({OUT.as_posix()}/射门比分状态测量说明.md) · '
            f'[研究接口]({ROOT.as_posix()}/model_v2/INTEGRATION.md)', '']
    if minute_path.exists():
        m = read(minute_path)
        lines += ['## 作者分钟包：已实际下载并联结', '',
            '借鉴的是公开作者工程里的状态、红牌与零事件分钟表示，已下载一个英超ZIP，'
            '未运行其R工程，也没有重新验证其论文成绩。包内2020/21、2021/22、2022/23'
            '分别376、354、369场，按原日期、显式球队别名和主客身份精确联结；'
            'ESPN ID不当作Understat ID，比分仅在联结后核对。', '',
            f"{m['total_package_games']:,}场全部联结到独立Football-Data及既有可信Understat记录；"
            f"{m['full_time_goal_matches']:,}场Goals事件合计吻合独立终场，"
            f"{m['half_time_goal_matches']:,}场吻合半场。唯一终场计数冲突是2022-09-03纽卡斯尔—水晶宫："
            '包内Goals累加1:0，两独立来源0:0。保留身份与异常记录，该场暂停用于状态修正；'
            '没有按终场比分修补事件，也不推定冲突原因。', '',
            f"{m['games_with_multiple_states_same_team_half_minute']}场有同队／同半场／同分钟多状态，"
            f"涉及{m['multiple_state_groups']}个键；不能一行当作1分钟。"
            '作者half_id、补时清洗和比分差不等于可靠秒级时钟或真实场上人数。'
            '事件总量核对通过也不能证明每个状态转移正确。', '',
            '**分钟包不含xG，且止于2022/23，与现有2023/24、2024/25的760场射门没有赛季重叠。**'
            '不能把两份资料拼接成同一场完整状态训练集，不能将整场xG摊到分钟，也不能给较新比赛凭空补红牌时点。', '',
            f'[精确联结说明]({OUT.as_posix()}/EPL分钟资料精确联结说明.md) · '
            f'[汇总及全部异常]({OUT.as_posix()}/EPL分钟资料精确联结.json)', '']
    lines += ['## 下一步的具体取舍', '',
        '先复用已有旧场预期，将历史攻防表现拆成“实际创造／允许机会−当时预计创造／允许机会”，'
        '再把攻防分量、总量、净差及波动作为一个联合信息包。'
        '这一步用已有资料即可推进，不必等待目标首发；它是在旧净残差均值之外补充历史结构，'
        '不是再次宣称首次加入对手强度，也不是纯粹增加变量数量。旧场预期可能有误差，残差不是因果能力。', '',
        '接入沿用现有预测器和已约定的时间切分，固定一次完整处理；'
        '不同时搜索新树设置、近期窗口、融合权重或按已知结果挑子组。'
        '分别报告原模型错误中的实际平局、主客方向反选，以及对数损失和改坏场次。'
        '一次接入后就总结所学与是否具备采用依据，不因分数没提高继续救分调参。', '',
        '完整比分／人数状态中和暂不拟合：缺口已具体到重叠赛季的机会事件、可靠阶段与状态时间暴露。'
        '只补“可匹配的真实信息”，不把局部早段射门或作者分钟清洗当成完整状态资料。'
        '是否值得补全，依可取得资料决定；这不妨碍先推进上述已有攻防信息的整合。', '',
        '本轮到错误定位、一个外部分钟包的精确联结和760场状态测量为止。'
        '新增训练0、参数搜索0、当前赛果查询0；统计／LLM／市场线路及既有比赛预测保持原记录。'
        '没有新增准确率结论，所有新增文件均在G盘。', '',
        f'[已有条件信息盘点]({OUT.as_posix()}/现有条件信息覆盖.md) · '
        f'[外部方法与复用边界]({OUT.as_posix()}/状态校正可复用方法与边界.md)', '']
    (OUT / '条件定位与下一步.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({period: value['all'] for period, value in summary.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

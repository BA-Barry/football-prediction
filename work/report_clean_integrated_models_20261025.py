"""Readable whole-scheme results; report every preset, no later winner swap."""
import argparse
from datetime import date
from pathlib import Path
import json

import numpy as np

from football_model.evaluation import calibration_bins
from run_shot_increment_20261025 import score_details, write


NAMES = {'glm_basic': '线性／基础', 'hgb_basic': '提升树／基础',
    'glm_performance': '线性／丰富表现', 'hgb_performance': '提升树／丰富表现',
    'extra_trees_performance': '随机树／丰富表现', 'glm_allcomp': '线性／表现＋全赛事',
    'hgb_allcomp': '提升树／表现＋全赛事', 'extra_trees_allcomp': '随机树／表现＋全赛事',
    'goal': '干净纯进球', 'xg_raw': '干净总xG', 'clean_blend': '干净进球／xG融合'}
LEAGUES = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
DOMAINS = [('exact_score', '完整比分'), ('one_x_two', '胜平负'),
           ('total_goals_0_1_2_3_4plus', '总进球'), ('away_minus_two_three_way', '客让两球')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def interval_row(test, key, title):
    index = test['difference_order'].index(key)
    intervals = []
    for weeks in ('2', '4', '8'):
        lower, upper = test['block_intervals'][weeks]['interval_95'][index]
        intervals.append(f'[{lower:+.6f}, {upper:+.6f}]')
    return f'| {title} | {test["mean_loss_differences"][key]:+.6f} | '+' | '.join(intervals)+' |'


def probability_diagnostics(out, result):
    tracks = {arm: [r for r in rows if r['period'] == '2025-26']
              for arm, rows in read(out/'逐场完整预测.json').items()}
    output = {'interpretation': 'descriptive probability checks only; no recalibration, subset selection or causal interpretation',
              'forecast_horizon_groups': {}, 'all_tracks': {}}
    selected = result['selected']
    for label, lower, upper in (('0-6days', 0, 6), ('7-13days', 7, 13), ('14-20days', 14, 20), ('21-27days', 21, 27)):
        local = {arm: [r for r in rows if lower <= (date.fromisoformat(r['date'])-
            date.fromisoformat(r['fit_before_date'])).days <= upper]
            for arm, rows in tracks.items()}
        output['forecast_horizon_groups'][label] = {'n': len(local['goal']),
            'scores': {arm: score_details(rows)[0] for arm, rows in local.items()}}
    for arm, rows in tracks.items():
        p = np.asarray([r['probabilities'] for r in rows])
        y = np.asarray([r['actual_outcome'] for r in rows])
        output['all_tracks'][arm] = {'n': len(rows), 'mean_probabilities': p.mean(axis=0).tolist(),
            'observed_frequencies': (np.bincount(y, minlength=3)/len(rows)).tolist(),
            'class_probability_bins': calibration_bins(p, y, bins=10)}
    write(out/'信息时点分组与概率校准诊断.json', output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    result = read(out/'完整方案结果汇总.json')
    data = read(out/'数据接入与覆盖.json')
    base = read(data['base_path'])
    selected, validation, test = result['selected'], result['validation'], result['test']
    probability = probability_diagnostics(out, result)
    key = f'{selected}_minus_clean_blend:exact_score'
    idx = test['difference_order'].index(key)
    consistent = all(item['interval_95'][idx][1] < 0 for item in test['block_intervals'].values())
    conclusion = ('较早选定方案的比分评分出现跨三种日历块长度的一致改善，保留为研究候选，仍不直接替换正式预测。'
        if consistent else '较早选定方案尚未显示稳定优于干净融合的证据；不采用替换，不按后季改选赢家。')
    lines = ['# 完整清理与全赛事联合预测方案', '', f'**本轮结论：{conclusion}**', '',
        '本次已经执行基础模型修复、历史特征重建和完整学习器重训；不同于上一轮只剔除评分。全部原始数据、旧参数和旧结果保留。当前赛果复核、首发和预计首发线路继续暂停，产物及临时文件均在G盘。', '',
        '## 本轮实际完成', '',
        f'- 基础专家428组中，{base["new_base_fits"]}组受新例外影响并实际重训，{base["reused_base_fits"]}组训练设计完全未变，经系数重放后复用。',
        f'- 独立联结后的历史表现源{data["source_matches"]:,}场，四季前推特征样本{result["rows"]:,}场，较早比较{validation["n"]:,}场、后季检验{test["n"]:,}场；全方案共同支持。',
        f'- 八套完整方案、{result["new_scheme_fits"]}次新拟合；每个得分侧基础10项、丰富表现120项、表现加全赛事160项。模型设置沿用，参数网格0个。',
        '- 新增赛程信息来自已下载2022/23—2025/26四季386个俱乐部赛季、10,012个去重历史赛事；只读取日期、赛事类别、状态及身份，不把杯赛比分或分钟当新训练目标。', '',
        '## 如何区分各部分的作用', '',
        '线性Poisson回归、梯度提升树和随机树分别接收同一丰富表现输入，再各增加同一40项全赛事输入。另有基础线性和基础提升树作参照。这是完整多变量方案及学习方法比较，不是逐个指标反复搜索权重。', '',
        '两种丰富输入都移除每队6个旧同联赛日期／场数变量，包含造成812天断档外推的旧间隔。两者保留相同历史表现、覆盖和预测提前天数；新增赛程线与表现线之间才可测新增赛程信息。不能把对旧坏输入的修复全部算成全赛事增益。', '',
        '丰富表现覆盖球队及对手近5／15场进失球、xG／NPxG、PPDA汇总分子分母与比值、深入进攻、变化趋势及相关主客场子样本。不足历史保留缺失及标志，不为了完整指标删掉比赛。', '',
        '全赛事输入包括截止日前7／14／28天正式赛事及非联赛数量、最近已确认正式赛到截止日的log1p间隔、近3场间隔和覆盖／状态提示。友谊赛单列不计正式负荷；已知中止、分段续赛或来源状态冲突进入不确定处理，不造90分钟。', '',
        '窗口严格相对原28天更新截止日，截止日及以后的比赛均不进入输入。预测提前天数两条丰富线共同持有；这些特征是当时已观察负荷，不是后来目标开球前实际最新负荷，也不是因果疲劳效应。', '',
        '## 时间推进与选择', '',
        '2022/23、2023/24前推样本启动新学习器训练。每个原28天截止只使用日期更早的比赛，并保留各历史样本自己的特征截止。2024/25八套比较后选择一次，2025/26继续利用更早已结束比赛更新参数，方案不再重选。没有随机切分比赛行。', '',
        f'**本轮较早选择：{NAMES[selected]}。** 选择属于本轮新数据／新输入研究，不能沿用上一轮方案的表现宣称同一检验；后季已被历史研究查看，本轮仍是回溯探索。', '',
        f'基础进球／xG融合权重从干净2024/25估计为{base["groups"]["recent"]["blend_parameters"]["weight"]:.6f}，固定用于2025/26。2024/25融合含该季拟合，未参与选择及较早评分。', '',
        '| 较早方案 | 比分NLL | 胜平负LL | 总球LL |', '|---|---:|---:|---:|']
    for arm, metrics in validation['scores'].items():
        lines.append(f'| {NAMES[arm]} | {metrics["exact_score"]["log_loss"]:.6f} | {metrics["one_x_two"]["log_loss"]:.6f} | {metrics["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} |')
    lines += ['', '## 后季全部方案', '',
        '损失越小越好。最高概率命中是三选一方向表现，不是概率正确率、收益率或中奖保证。', '',
        '| 方案 | 比分NLL | 胜平负LL | 胜平负Brier | 总球LL | 最高概率命中 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, metrics in test['scores'].items():
        lines.append(f'| {NAMES[arm]} | {metrics["exact_score"]["log_loss"]:.6f} | {metrics["one_x_two"]["log_loss"]:.6f} | {metrics["one_x_two"]["brier_sum"]:.6f} | {metrics["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} | {round(metrics["one_x_two"]["accuracy"]*test["n"])}/{test["n"]} |')
    lines += ['', '### 较早选择减干净融合', '',
        '| 指标 | 损失差 | 2周95%条件区间 | 4周95%条件区间 | 8周95%条件区间 |',
        '|---|---:|---|---|---|']
    for domain, label in DOMAINS:
        lines.append(interval_row(test, f'{selected}_minus_clean_blend:{domain}', label))
    lines += ['', '### 新赛程信息的同学习器对照', '',
        '下表是表现加全赛事减同学习器丰富表现，保留全部三个学习器，不能据后季挑一个赛程赢家。', '',
        '| 学习器／指标 | 损失差 | 2周95%条件区间 | 4周95%条件区间 | 8周95%条件区间 |',
        '|---|---:|---|---|---|']
    for kind, title in (('glm', '线性'), ('hgb', '提升树'), ('extra_trees', '随机树')):
        for domain, label in DOMAINS[:3]:
            lines.append(interval_row(test, f'{kind}_allcomp_minus_{kind}_performance:{domain}', f'{title}／{label}'))
    lines += ['', '### 跨联赛表现', '',
        '| 联赛 | 场数 | 较早选择比分NLL | 干净融合比分NLL | 差值 |', '|---|---:|---:|---:|---:|']
    for comp, section in result['by_league'].items():
        lines.append(f'| {LEAGUES[comp]} | {section["n"]} | {section["scores"][selected]["exact_score"]["log_loss"]:.6f} | {section["scores"]["clean_blend"]["exact_score"]["log_loss"]:.6f} | {section["mean_loss_differences"][key]:+.6f} |')
    lines += ['', '### 概率本身及信息新旧分组', '',
        '均值只反映整体偏差，不能代替逐场概率校准。各概率区间和全部线路均保留于诊断文件；本轮未据此做新校准或选子集。', '',
        '| 类别 | 较早选择平均概率 | 干净融合平均概率 | 真实频率 |', '|---|---:|---:|---:|']
    for i, label in enumerate(('主胜', '平局', '客胜')):
        s = probability['all_tracks'][selected]
        b = probability['all_tracks']['clean_blend']
        lines.append(f'| {label} | {s["mean_probabilities"][i]:.3%} | {b["mean_probabilities"][i]:.3%} | {s["observed_frequencies"][i]:.3%} |')
    lines += ['', '| 截止至目标日 | 场数 | 较早选择比分NLL | 干净融合比分NLL | 差值 |', '|---|---:|---:|---:|---:|']
    for label, item in probability['forecast_horizon_groups'].items():
        s, b = item['scores'][selected]['exact_score']['log_loss'], item['scores']['clean_blend']['exact_score']['log_loss']
        lines.append(f'| {label} | {item["n"]} | {s:.6f} | {b:.6f} | {s-b:+.6f} |')
    timing = read(out/'信息时点与负荷覆盖诊断.json')['overall']
    stale_fraction = timing['team_contexts_with_excluded_between_events']/timing['team_contexts']
    lines += ['', '## 信息时点：已经量化的下一项问题', '',
        f'按既定28天截止，预测提前天数中位{timing["forecast_lead_days"]["median"]:g}天，最近已知正式赛距目标开球的日期间隔中位{timing["latest_usable_before_cutoff_gap_to_target_days"]["median"]:g}天、90分位{timing["latest_usable_before_cutoff_gap_to_target_days"]["p90"]:g}天。{timing["team_contexts_with_excluded_between_events"]:,}/{timing["team_contexts"]:,}队次（{stale_fraction:.1%}）在截止日以后、目标日以前已有正式比赛，合计{timing["excluded_between_cutoff_and_target_formal_events_total"]:,}次队次事件引用被旧规则正确排除。', '',
        '四季截止日固定落在周一、周二、周四、周五，星期与赛季完全混杂。最近已知正式赛是非联赛的比例随季改变，不能直接解释为球队负荷变化或预测价值变化。所有诊断使用回溯日期作覆盖测量，没有把截止日后的比赛补进本轮特征。', '',
        '因此本轮只能检验旧信息更新时点下的增量。下一项将参数重训频率与输入刷新分开，构建统一赛前1日／3日的日期级历史快照；学习器参数仍只在原周期按更早目标更新，但每个历史训练例子也必须使用相同提前规则。参数日期还必须不晚于该预测发布日，不能沿用虽早于比赛、却晚于发布日的旧参数。保留本轮各线作参照，不根据这些事后日期差宣称刷新必然增准。精确赛前24／72小时仍需可信开球与发布时间，当前日期级资料不足以声称做到。', '',
        '旧同联赛两项赛程模型已经尝试过“参数周期更新、预测时更新日期特征”，未发现稳定增益；分开更新本身不是新发明。下一项的区别是固定发布提前日期、丰富表现与全赛事统一快照，以及基础专家和新学习器都遵守发布日，不能把旧失败路线重新命名成已证明有效的改进。', '',
        '## 采用与后续安排', '', conclusion, '',
        '本轮修正可直接保留的是数据语义：完整比分目标排除已登记不完整赛事、资料缺口不冒充实际休息、全赛事日期与覆盖分别表达。预测增益则需以全部方案及不确定性评估，不能因接入更多字段便自动替换模型。', '',
        '如果赛程线未稳定改善，停止围绕7／14／28窗口或树深继续小调。先完成上述输入时点的完整前推，随后再研究球队实力如何随比赛持续更新，以及现有主客独立Poisson对总进球／平局的系统性偏差。先说明现有方案重复失败的位置，再讨论结构调整；需要神经网络时再与用户讨论。', '',
        '统计、独立xG、市场、LLM与联合线路保留。LLM后续使用可核时间的公告和赛事情境，不根据历史赛果回写赛前看法；市场缺历史早期观察时间时继续作参考，不能作为与早期模型同信息时点的输入。', '',
        '## 核验与适用边界', '',
        '196项相关测试通过。176组模型903,272条训练引用、两季21条评分线路的四类概率评分独立重算通过，最大评分差6.31×10⁻¹⁴；48场保存模型率值重放最大差2.45×10⁻¹⁵。基础专家453,576条引用、207,011条表现引用及573,016项赛程／提前量特征另行核对通过。训练、插补、标准化均仅使用更早资料，保留全部合格目标；这不等于来源完整性或精确公告时间已经得到证明。', '',
        '原Cremonese—Parma丰富线性方案曾输出53.87／0.075球；本轮丰富表现线性为1.199／1.131，全赛事线性为0.820／1.761，较早选定随机树为1.217／1.227。八方案在后季的最高单侧均值为3.26—4.60，均未超过5球。原资料断档导致的极端数值失败已消除，不能据个别样本把新均值称为准确，或将全部评分变化归因于一个字段。', '',
        '三类算法仍输出主客条件进球均值，再共同使用独立Poisson比分分布。随机树不是所有模型不确定性的概率混合；本轮未解决全部过离散、双方相关性及参数误差。国内联赛球队尺度也不能直接证明欧战跨联赛预测成熟。', '',
        '历史xG与赛程的首次发布时间、修订、延期公告时点仍未知；仅能保证现存记录中的日期更早。来源标记已完成也不等于可靠踢满90分钟，清除当前登记例外不代表已穷尽全部未知异常。日历块区间仅条件于当前模型及研究设计，不涵盖全部选择与参数不确定性。', '',
        f'[基础专家重建说明]({Path(data["base_path"]).resolve().parent.as_posix()}/基础专家重建说明.md) · '
        f'[完整评分]({(out/"完整方案结果汇总.json").as_posix()}) · '
        f'[全部逐场概率]({(out/"逐场完整预测.json").as_posix()}) · '
        f'[独立输入核对]({(out/"独立完整输入核对.json").as_posix()}) · '
        f'[模型及概率评分核对]({(out/"独立模型与全概率评分核对.json").as_posix()}) · '
        f'[旧极端均值修复检查]({(out/"旧极端样本与新方案均值检查.json").as_posix()}) · '
        f'[下一轮输入刷新设计]({(out/"输入刷新前推设计.md").as_posix()})', '']
    target = out/'完整重训结论与下一步.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    main()

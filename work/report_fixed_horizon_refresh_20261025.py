"""Report the completed fixed-publication comparison without later reselection.

The study itself writes all fixed alternatives. This script summarizes actual
saved fits and probabilities; it never trains, calibrates, or changes forecasts.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
NAMES = {'fresh_performance': '刷新表现', 'fresh_allcomp': '刷新表现＋全赛事',
         'locked_allcomp': '锁定表现＋全赛事', 'goal': '合法较早纯进球',
         'xg_raw': '合法较早总xG', 'clean_blend': '较早学习进球／xG融合'}
LEAGUES = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
DOMAINS = [('exact_score', '完整比分'), ('one_x_two', '胜平负'),
           ('total_goals_0_1_2_3_4plus', '总进球'), ('away_minus_two_three_way', '客让两球')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def interval_values(comparison, contrast, domain):
    key = f'{contrast}:{domain}'
    index = comparison['difference_order'].index(key)
    return comparison['mean_loss_differences'][key], {
        weeks: comparison['block_intervals'][weeks]['interval_95'][index]
        for weeks in ('2', '4', '8')}


def interval_line(comparison, contrast, domain, label):
    difference, intervals = interval_values(comparison, contrast, domain)
    return f'| {label} | {difference:+.6f} | '+' | '.join(
        f'[{intervals[w][0]:+.6f}, {intervals[w][1]:+.6f}]' for w in ('2', '4', '8'))+' |'


def pattern_conclusion(result):
    comparison = 'fresh_allcomp_minus_locked_allcomp'
    values = {lead: interval_values(item['test'], comparison, 'exact_score')
              for lead, item in result['horizons'].items()}
    means = [value[0] for value in values.values()]
    all_upper_negative = all(bounds[1] < 0 for _, intervals in values.values() for bounds in intervals.values())
    all_lower_positive = all(bounds[0] > 0 for _, intervals in values.values() for bounds in intervals.values())
    if all_upper_negative:
        conclusion = ('固定提前日期下，刷新整套输入相对锁定输入的比分评分在两档提前量中均改善，'
                      '三种日历块区间也一致低于零。可保留为进一步验证的研究方案，原赛前预测不自动替换。')
    elif all_lower_positive:
        conclusion = ('固定提前日期下，当前整套输入刷新方案相对锁定输入在比分评分上变差，'
                      '两档提前量的日历块区间也一致支持这一方向。不采用此刷新方案，停止围绕窗口小调。')
    elif all(value < 0 for value in means):
        conclusion = ('刷新整套输入的平均比分损失在两档提前量中均下降，但不确定区间未一致排除零，'
                      '尚不足以确认稳定增益；保留记录，不替换原赛前预测。')
    elif all(value >= 0 for value in means):
        conclusion = ('刷新整套输入未改善两档提前量的平均比分评分，目前没有采用它替换锁定输入的依据。'
                      '不通过另挑提前量、窗口或少数命中场次追分。')
    else:
        conclusion = ('刷新整套输入的比分变化在两档提前量间方向不一致，尚无稳定增益证据。'
                      '不按后季成绩选择一个提前量，不替换原赛前预测。')
    return conclusion


def metric_table(lines, scores, n):
    lines.extend(['| 方案 | 比分NLL | 胜平负LL | 胜平负Brier | 总球LL | 方向命中 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for arm, metrics in scores.items():
        lines.append(f'| {NAMES[arm]} | {metrics["exact_score"]["log_loss"]:.6f} | '
            f'{metrics["one_x_two"]["log_loss"]:.6f} | {metrics["one_x_two"]["brier_sum"]:.6f} | '
            f'{metrics["total_goals_0_1_2_3_4plus"]["log_loss"]:.6f} | '
            f'{round(metrics["one_x_two"]["accuracy"]*n)}/{n} |')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    result = read(out/'固定发布日期完整汇总.json')
    plan = result['plan']
    base_path = Path(plan['base_path'])
    base = read(base_path)
    finished = 0
    for lead, summary in result['horizons'].items():
        folder = out/f'lead_{lead}'
        parameter_count = len(read(folder/'全部模型参数.json'))
        saved_count = len(list((folder/'models').glob('*.joblib')))
        if parameter_count != saved_count or parameter_count != summary['new_scheme_fits']:
            raise ValueError('Readable report requires the actual completed fit files, not the planned count')
        finished += parameter_count
    if finished != result['new_scheme_fits']:
        raise ValueError('Actual completed horizon fits differ from root summary')
    test_log = ROOT/'work/fixed_horizon_tests_20261025.log'
    test_count = None
    if test_log.exists():
        matches = re.findall(r'(\d+) passed', test_log.read_text(encoding='utf-8'))
        if matches:
            test_count = int(matches[-1])
    conclusion = pattern_conclusion(result)
    lines = ['# 固定发布日期下的信息刷新：完整实跑结果', '', f'**结论：{conclusion}**', '',
        f'本轮实际完成{finished}次随机树完整拟合，而非只更改预测日期或复制上一轮模型。'
        '每个提前量保留刷新表现、刷新表现加全赛事、锁定表现加全赛事三套固定方案，并保留纯进球、总xG和较早学习融合参照。'
        '原赛前预测及市场、LLM线路未改写；首发和当前赛果复核仍暂停，所有产物和临时文件在G盘。', '',
        '## 验证的问题与时间规则', '',
        '本轮检验整套较新信息能否改善预测，并区分新增全赛事信息。两档提前量是不同使用时点，全部报告，不按后季成绩挑选赢家。'
        '算法继续使用相同ExtraTrees设置：120树、最多6层、叶至少60样本、每次随机取80%特征、180天半衰期；未搜索窗口、树深或融合网格。', '',
        '目标日为D，发布日F＝D−1或D−3，参数日P为该赛季8月1日锚定28天中不晚于F的最近日期。'
        '基础专家、新随机树及融合学习时点都不晚于F。刷新方案仅使用比赛日严格早于F的历史，锁定方案仅使用严格早于P的历史；各历史训练例遵守其自身相同提前规则。', '',
        '两刷新方案共享121项表现、基础球队率和时间输入，全赛事方案再增加相同40项赛程输入；锁定全赛事方案有161项同定义输入，'
        '在更早参数日形成。日历级差异不等于精确24／72小时，历史首次公布、xG修订与改期公告时间仍未知。', '',
        '2022/23与2023/24前推样本启动随机树训练。2024/25保留全部方案比较，2025/26按原周期继续只用更早目标更新。'
        '没有按较早或较晚成绩再选最佳方案／提前量；后季此前已被查看，本轮仍属回溯研究，不是新的盲测。', '',
        '## 覆盖与实际工作量', '',
        f'基础球队专家{base["reused_base_fits"]}组按完整较早训练设计复用，新增基础拟合{base["new_base_fits"]}次。'
        '发布时间更早造成部分季初球队尚未被该合法模型观察，必须明确不可预测，不能把缺少系数当作已知影响为0。', '',
        '| 提前量 | 四季可预测 | 明确不可预测 | 较早比较2024/25 | 后季2025/26 | 参数切到更早版本 | 新完整拟合 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for lead, summary in result['horizons'].items():
        cover = result['coverage'][lead]
        base_cover = base['horizons'][lead]['coverage']
        lines.append(f'| 前{lead}日 | {cover["n"]:,} | {len(cover["unavailable"])} | '
            f'{summary["validation"]["n"]:,} | {summary["test"]["n"]:,} | '
            f'{base_cover["shifted_parameter_n"]} | {summary["new_scheme_fits"]} |')
    lines += ['', '| 提前量 | 刷新表现不足历史 | 刷新全赛事不足历史 | 锁定全赛事不足历史 |', '|---|---:|---:|---:|']
    for lead, cover in result['coverage'].items():
        counts = cover['diagnostic_counts']
        lines.append(f'| 前{lead}日 | {counts.get("fresh_performance_insufficient_history",0)} | '
            f'{counts.get("fresh_allcomp_insufficient_history",0)} | {counts.get("locked_allcomp_insufficient_history",0)} |')
    lines += ['', '不足5场的表现历史仍保留目标、基础均值与可用标记；赛程窗口覆盖不足、部分比赛或状态冲突保留缺失。'
        '未知基础球队的记录另列在不可预测清单。两提前量覆盖略不同，不能直接把总体均分差解释成提前日期效果；每档内部全部方案严格使用相同目标。', '']
    for lead, summary in result['horizons'].items():
        validation, test = summary['validation'], summary['test']
        lines += [f'## 比赛前{lead}日：全部方案', '', '损失越低越好，Brier为三类平方误差之和。'
            '方向命中仅是取最高概率类别后的三选一表现，不等于概率正确率、盈利或中奖保证。', '',
            '### 较早赛季', '']
        metric_table(lines, validation['scores'], validation['n'])
        lines += ['', '较早融合权重在本季学习，属拟合内描述，未放入这张较早比较表。', '', '### 后季', '']
        metric_table(lines, test['scores'], test['n'])
        lines += ['', '### 主比较：刷新整套输入减锁定整套输入', '',
            '| 指标 | 损失差 | 2周95%条件区间 | 4周95%条件区间 | 8周95%条件区间 |',
            '|---|---:|---|---|---|']
        for domain, title in DOMAINS:
            lines.append(interval_line(test, 'fresh_allcomp_minus_locked_allcomp', domain, title))
        lines += ['', '### 同时保留的新赛程信息及融合参照', '',
            '| 比较／指标 | 损失差 | 2周95%条件区间 | 4周95%条件区间 | 8周95%条件区间 |',
            '|---|---:|---|---|---|']
        for contrast, title in [('fresh_allcomp_minus_fresh_performance', '刷新全赛事减刷新表现'),
                                ('fresh_allcomp_minus_clean_blend', '刷新全赛事减较早融合'),
                                ('fresh_performance_minus_clean_blend', '刷新表现减较早融合')]:
            for domain, label in DOMAINS[:3]:
                lines.append(interval_line(test, contrast, domain, f'{title}／{label}'))
        lines += ['', '### 五联赛主比较', '',
            '| 联赛 | 场数 | 刷新比分NLL | 锁定比分NLL | 差值 |', '|---|---:|---:|---:|---:|']
        for comp, section in summary['by_league'].items():
            fresh = section['scores']['fresh_allcomp']['exact_score']['log_loss']
            locked = section['scores']['locked_allcomp']['exact_score']['log_loss']
            lines.append(f'| {LEAGUES[comp]} | {section["n"]} | {fresh:.6f} | {locked:.6f} | {fresh-locked:+.6f} |')
        group = base['horizons'][lead]['blend_parameters']['recent']
        lines += ['', f'该档较早2024/25学得xG融合权重为{group["weight"]:.6f}，只固定用于后季参照；'
            '不是因为后季表现而挑出的比例。', '']
    lines += ['## 当前采用与后续', '', conclusion, '',
        '本轮直接保留的是合法发布时间、参数日期与输入日期分开、未知球队和覆盖不足不填造的实现。'
        '新增信息的预测价值则按两档全部结果和不确定性评估，不因信息更近、字段更多或命中场数增加而自动采用。', '',
        '若两档和不同输出未形成一致增益，不继续小调窗口、提前日期、树深或按联赛挑参数。'
        '本轮两档总进球分组损失点估计均变差；前3日的2／4／8周条件区间均在零以上，应一并保留，不能只摘胜平负的小幅下降作为采用依据。', '',
        '用户已明确同意下一项新增独立胜平负分类对照：在同一发布日、输入与比赛集合上，直接学习主胜／平局／客胜三类概率，'
        '以多项逻辑回归及固定复杂度分类树研究“先拟合两个进球均值再经独立Poisson转换”是否限制胜平负表现。'
        '当前进球、比分、总球及LLM线路保留；分类概率不能自动反推出完整比分或让球，也不加入神经网络、不直接上线。'
        '本报告仅完成刷新研究，不把新分类线提前称为训练完成或有效。当前首发研究仍暂停。', '',
        '## 核验及解释边界', '']
    if test_count is not None:
        lines.append(f'{test_count}项联合相关测试通过，数量取自本轮实际测试日志。')
        lines.append('')
    for lead in result['horizons']:
        for label, name in [('完整输入', f'独立特征核对_lead{lead}.json'),
                            ('模型及概率评分', f'独立模型与概率评分核对_lead{lead}.json')]:
            path = out/name
            if path.exists():
                audit = read(path)
                status = audit.get('status', 'unknown')
                detail = ''
                if status == 'passed' and label == '模型及概率评分':
                    detail = f'，{audit.get("saved_models","未知")}组保存模型、{audit.get("training_reference_entries","未知")}条训练引用'
                    detail += f'，最大评分差{audit.get("max_score_error", float("nan")):.3g}'
                lines.append(f'- 前{lead}日{label}核对：{status}{detail}。[核对文件]({path.as_posix()})')
            else:
                lines.append(f'- 前{lead}日{label}独立核对尚未提供，本报告不把缺失核对称为通过。')
    lines += ['', '共同独立Poisson比分分布仍以主客条件均值为基础，未自动解决过离散、双方相关、参数误差或全部模型不确定性。'
        '日历块区间仅条件于本轮估计模型及固定研究设计，不涵盖全部研究选择、资料修订和潜在源遗漏。'
        '比分改善、方向命中增加与市场价值或收益不是同一个指标。国内联赛尺度也不能直接证明欧战跨联赛预测已成熟。', '',
        f'[全部汇总]({(out/"固定发布日期完整汇总.json").as_posix()}) · '
        f'[覆盖与不可预测清单]({(out/"覆盖与不可预测.json").as_posix()}) · '
        f'[基础率及合法参数说明]({(base_path.parent/"固定发布日期基础率说明.md").as_posix()})', '']
    target = out/'固定发布日期实跑结论与后续.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    main()

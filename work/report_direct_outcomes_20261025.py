"""Readable, file-derived report for the completed direct-1X2 comparison.

This script only reads saved study outputs. It does not import the learners,
fit anything, choose a winning alternative, or overwrite match forecasts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT
import re


ROOT = REPOSITORY_ROOT
NAMES = {
    'logistic_performance': '多项逻辑回归／表现',
    'logistic_allcomp': '多项逻辑回归／表现＋全赛事',
    'extra_trees_performance': '直接分类树／表现',
    'extra_trees_allcomp': '直接分类树／表现＋全赛事',
    'regression_performance': '进球回归转胜平负／表现',
    'regression_allcomp': '进球回归转胜平负／表现＋全赛事',
    'goal': '较早纯进球专家',
    'xg_raw': '较早独立总xG专家',
    'clean_blend': '较早学得的进球／xG融合',
}
LEAGUES = {'E0': '英超', 'D1': '德甲', 'SP1': '西甲', 'I1': '意甲', 'F1': '法甲'}
CLASSES = {'home': '主胜', 'draw': '平局', 'away': '客胜'}
METRICS = {'log_loss': '胜平负LL', 'brier_sum': 'Brier和', 'rps': 'RPS',
           'misclassification': '方向错误率'}
PRIMARY = 'extra_trees_allcomp_minus_regression_allcomp'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def label(arm):
    return NAMES.get(arm, arm)


def signed(value):
    # Seven digits retain the sign of very small conditional interval bounds.
    return f'{value:+.7f}'


def percentage(value):
    return f'{value * 100:.2f}%'


def contrast_label(contrast):
    arm, reference = contrast.split('_minus_', 1)
    return f'{label(arm)} 减 {label(reference)}'


def interval_values(section, contrast, metric):
    key = f'{contrast}:{metric}'
    index = section['difference_order'].index(key)
    intervals = {weeks: value['interval_95'][index]
                 for weeks, value in section['block_intervals'].items()}
    return section['mean_loss_differences'][key], intervals


def direction_text(values, metric_name):
    """Describe all retained horizons; this is not an adoption threshold."""
    means = [mean for mean, _ in values]
    bounds = [pair for _, intervals in values for pair in intervals.values()]
    if all(value < 0 for value in means):
        if bounds and all(pair[1] < 0 for pair in bounds):
            return f'{metric_name}在全部提前量中下降，已保存的各日历块区间也均低于零'
        return f'{metric_name}在全部提前量的点估计下降，但日历块区间未一致排除零'
    if all(value >= 0 for value in means):
        if bounds and all(pair[0] > 0 for pair in bounds):
            return f'{metric_name}在全部提前量中增加，已保存的各日历块区间也均高于零'
        return f'{metric_name}在全部提前量的点估计均未改善，日历块区间未一致排除零'
    return f'{metric_name}在不同提前量间方向不一致'


def conclusion(result):
    sections = [item['test'] for item in result['horizons'].values()]
    ll = [interval_values(item, PRIMARY, 'log_loss') for item in sections]
    brier = [interval_values(item, PRIMARY, 'brier_sum') for item in sections]
    rps = [interval_values(item, PRIMARY, 'rps') for item in sections]
    decision = '本轮不采用替换。' if result.get('adopted') is False else '采用状态以完整汇总中的实际记录为准。'
    return (f'主比较中，{direction_text(ll, "胜平负LL")}；'
            f'{direction_text(brier, "Brier和")}；{direction_text(rps, "RPS")}。'
            + decision + '结论限于本次固定配置，不能泛化为全部直接分类算法无效。'
            '分类方案的完整结论还须结合下列逐类别、逐概率段校准和逻辑回归对照，'
            '不能由方向命中率或整体平局均值单独决定。')


def metric_table(lines, section):
    lines += ['| 方案 | 场数 | 胜平负LL | Brier和 | RPS | 方向命中 |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, score in section['scores'].items():
        n = score['n']
        lines.append(f'| {label(arm)} | {n:,} | {score["log_loss"]:.7f} | '
                     f'{score["brier_sum"]:.7f} | {score["rps"]:.7f} | '
                     f'{round(score["accuracy"] * n)}/{n}（{percentage(score["accuracy"])}） |')


def primary_overview(result):
    lines = ['| 同场主比较 | 场数 | 分类树LL | 进球回归LL | LL差 | 4周95%条件区间 | Brier差 |',
             '|---|---:|---:|---:|---:|---|---:|']
    for lead, item in result['horizons'].items():
        test = item['test']
        delta, intervals = interval_values(test, PRIMARY, 'log_loss')
        brier = interval_values(test, PRIMARY, 'brier_sum')[0]
        bounds = intervals['4']
        lines.append(f'| 前{lead}日 | {test["n"]:,} | '
                     f'{test["scores"]["extra_trees_allcomp"]["log_loss"]:.7f} | '
                     f'{test["scores"]["regression_allcomp"]["log_loss"]:.7f} | {signed(delta)} | '
                     f'[{signed(bounds[0])}, {signed(bounds[1])}] | {signed(brier)} |')
    return lines + ['']


def interval_table(lines, section, contrasts):
    weeks = list(section['block_intervals'])
    lines += ['| 比较／指标 | 损失差 | ' + ' | '.join(f'{w}周95%条件区间' for w in weeks) + ' |',
              '|---|---:|' + '|'.join('---' for _ in weeks) + '|']
    for contrast in contrasts:
        for metric in METRICS:
            if f'{contrast}:{metric}' not in section['mean_loss_differences']:
                continue
            mean, intervals = interval_values(section, contrast, metric)
            lines.append(f'| {contrast_label(contrast)}／{METRICS[metric]} | {signed(mean)} | '
                         + ' | '.join(f'[{signed(intervals[w][0])}, {signed(intervals[w][1])}]'
                                      for w in weeks) + ' |')


def class_diagnostic(score, class_name, index):
    bins = [item for item in score['class_probability_bins'] if item['outcome'] == class_name]
    n = sum(item['n'] for item in bins)
    if n != score['n']:
        raise ValueError('Saved calibration bins must account for every fixture in each class')
    mean = score['mean_probabilities'][index]
    frequency = score['observed_frequencies'][index]
    weighted_gap = sum(item['n'] * abs(item['mean_probability'] - item['observed_frequency'])
                       for item in bins) / n
    actual = score['by_actual_class'][class_name]
    return mean, frequency, weighted_gap, len(bins), actual


def calibration_table(lines, section, class_order):
    lines += ['| 方案／类别 | 平均概率 | 实际比例 | 平均概率减实际 | 分箱加权绝对差 | 实际该类场数／LL |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, score in section['scores'].items():
        for i, class_name in enumerate(class_order):
            mean, frequency, gap, _, actual = class_diagnostic(score, class_name, i)
            lines.append(f'| {label(arm)}／{CLASSES.get(class_name, class_name)} | '
                         f'{percentage(mean)} | {percentage(frequency)} | '
                         f'{(mean - frequency) * 100:+.2f}百分点 | {gap * 100:.2f}百分点 | '
                         f'{actual["n"]}／{actual["log_loss"]:.6f} |')


def calibration_summary(result):
    parts, changes = [], []
    classes = result['plan']['class_order']
    for lead, item in result['horizons'].items():
        scores = item['test']['scores']
        direct, reference = scores['extra_trees_allcomp'], scores['regression_allcomp']
        newer, older = [], []
        for i, class_name in enumerate(classes):
            new_gap = class_diagnostic(direct, class_name, i)[2]
            old_gap = class_diagnostic(reference, class_name, i)[2]
            newer.append(new_gap)
            older.append(old_gap)
            changes.append(new_gap - old_gap)
        parts.append(f'前{lead}日的分类树分箱加权绝对差依次为'
                     + '／'.join(f'{gap * 100:.2f}' for gap in newer)
                     + '个百分点，对应进球回归为'
                     + '／'.join(f'{gap * 100:.2f}' for gap in older) + '个百分点')
    order = '／'.join(CLASSES.get(value, value) for value in classes)
    qualitative = ('这些已保存分箱的描述量均下降' if all(value < 0 for value in changes)
                   else '这些已保存分箱的描述量并未全部改善')
    return (f'按{order}顺序，' + '；'.join(parts) + '。' + qualitative
            + '，说明类别概率的可靠性形态发生了变化；但分箱描述没有独立区间，'
            '仍不能取代主概率损失及其不确定性，也不是对所有概率范围可靠性的证明。')


def calibration_detail(result, target):
    lines = ['# 直接胜平负研究：全部已保存概率分箱', '',
             '以下完整保留全部方案、提前量和两个赛季的非空10等宽概率段；没有挑选表现好的类别或箱段。'
             '每个类别独立按其预测概率分箱，三类会分别计入同一批比赛；各类箱段场数合计等于该方案比赛数。'
             '低场数箱段波动较大，本表未提供其置信区间，也未用于后验校准。', '']
    for lead, item in result['horizons'].items():
        for period_name, period in [('较早赛季', 'validation'), ('后季', 'test')]:
            section = item[period]
            lines += [f'## 前{lead}日／{period_name}', '',
                      '| 方案 | 类别 | 概率段 | 场数 | 平均预测概率 | 实际比例 | 预测减实际 |',
                      '|---|---|---|---:|---:|---:|---:|']
            for arm, score in section['scores'].items():
                for value in score['class_probability_bins']:
                    b = value['bin']
                    close = ']' if b == 9 else ')'
                    gap = value['mean_probability'] - value['observed_frequency']
                    lines.append(f'| {label(arm)} | {CLASSES.get(value["outcome"], value["outcome"])} | '
                                 f'[{b / 10:.1f}, {(b + 1) / 10:.1f}{close} | {value["n"]} | '
                                 f'{percentage(value["mean_probability"])} | '
                                 f'{percentage(value["observed_frequency"])} | {gap * 100:+.2f}百分点 |')
            lines.append('')
    target.write_text('\n'.join(lines), encoding='utf-8')


def find_statuses(payload):
    if isinstance(payload, dict):
        if 'status' in payload:
            yield str(payload['status'])
        for key, value in payload.items():
            if key != 'status' and isinstance(value, (list, dict)):
                yield from find_statuses(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from find_statuses(value)


def actual_fits(out, result):
    counts = {}
    for lead, summary in result['horizons'].items():
        folder = out / f'lead_{lead}'
        parameters = read(folder / '全部分类模型参数.json')
        saved = len(list((folder / 'models').glob('*.joblib')))
        if len(parameters) != saved or saved != summary['new_classifier_fits']:
            raise ValueError('Report requires matching actual parameter records, saved models and completed fit count')
        arm_counts = Counter(key.split(':', 1)[0] for key in parameters)
        cutoffs = {value['before_date'] for value in parameters.values()}
        feature_counts = {}
        for value in parameters.values():
            family = value['pipeline_feature_set']
            feature_counts.setdefault(family, set()).add(len(value['feature_names']))
        counts[lead] = {'n': saved, 'arm_counts': arm_counts, 'cutoffs': len(cutoffs),
                        'feature_counts': feature_counts}
    if sum(value['n'] for value in counts.values()) != result['new_classifier_fits']:
        raise ValueError('Completed root fit count differs from saved horizon models')
    return counts


def audit_table(lines, out, source):
    lines += ['## 实际核验状态', '']
    log = ROOT / 'work' / 'direct_outcomes_tests_20261025.log'
    if log.exists():
        values = re.findall(r'(\d+) passed', log.read_text(encoding='utf-8'))
        if values:
            lines += [f'相关联合测试实际通过{values[-1]}项，数量取自[本轮测试日志]({log.as_posix()})。', '']
        else:
            lines += ['测试日志尚无完整通过数量，本报告不把它解释为全部通过。', '']
    else:
        lines += ['本轮测试日志尚未提供。', '']
    audits = sorted(out.glob('独立*.json'))
    if not audits:
        lines += ['- 本轮独立审计文件尚未提供，不把上轮审计替代为本轮已通过。']
    for path in audits:
        statuses = list(find_statuses(read(path)))
        status = '、'.join(dict.fromkeys(statuses)) if statuses else '文件已提供，但没有明确status字段'
        lines.append(f'- [{path.stem}]({path.as_posix()})：{status}。')
    total_path = out / '独立直接分类总核对.json'
    if total_path.exists():
        total = read(total_path)
        if total.get('status') == 'passed' and total.get('audits'):
            parts = total['audits']
            references = sum(value['training_reference_entries'] for value in parts)
            replays = sum(value['all_classifier_prediction_replays'] for value in parts)
            tracks = sum(value['score_tracks_recomputed'] for value in parts)
            bins = sum(value['calibration_bins_recomputed'] for value in parts)
            model_error = max(value['max_estimator_replay_error'] for value in parts)
            independent_error = max(value['max_independent_serialized_computation_error'] for value in parts)
            score_error = max(value['max_score_error'] for value in parts)
            lines += ['', f'总核对实际覆盖{total["saved_classifier_models"]}个分类模型、'
                      f'{references:,}条跨参数日期／方案重复的更早训练引用、{replays:,}条分类预测，'
                      f'以及{tracks}个评分范围和{bins:,}个概率分箱。'
                      f'保存模型回放最大差{model_error:.3g}，独立按序列化系数／树叶重算最大差{independent_error:.3g}，'
                      f'评分最大差{score_error:.3g}。这些引用和预测条目均不等于独立比赛数量。', '']
    source_audits = sorted(source.glob('独立*.json'))
    if source_audits:
        lines += ['', '本轮输入沿用上轮信息刷新研究；其原始输入重算及进球线审计另列，不能代替本轮分类模型审计：', '']
        for path in source_audits:
            statuses = list(find_statuses(read(path)))
            status = '、'.join(dict.fromkeys(statuses)) if statuses else '未提供明确status字段'
            lines.append(f'- [{path.stem}]({path.as_posix()})：{status}。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    result = read(out / '直接胜平负完整汇总.json')
    plan = result['plan']
    source = Path(plan['source_study'])
    coverage = read(out / '覆盖与不可预测.json')
    counts = actual_fits(out, result)
    finished = sum(value['n'] for value in counts.values())
    class_order = plan['class_order']
    detail = out / '逐类别概率分箱全表.md'
    calibration_detail(result, detail)
    final_conclusion = conclusion(result)
    calibration_interpretation = calibration_summary(result)
    lines = ['# 直接学习胜平负：固定方案完整实跑结果', '',
             f'**结论：{final_conclusion}**', '', calibration_interpretation, '',
             f'本轮实际完成{finished}次分类模型拟合，数量来自保存模型、参数与完成汇总。'
             '它研究的是预测问题本身：相同赛前信息直接学习主胜／平局／客胜，'
             '与先学习双方进球均值、再经独立Poisson转换胜平负作比较。'
             '原进球、比分、总球、市场及LLM线路保留，当前首发和赛果复核仍暂停。', '',
             '## 问题、算法与公平比较的范围', '',
             '每场比赛只贡献一行和一个三分类标签。主队视角同时包含主、客两队的表现、基础率和时间信息；'
             '客队视角是角色交换，不额外重复成第二个训练标签。两组输入分别为表现输入，以及表现加全赛事赛程输入；'
             '与对应进球回归线在同一提前量、发布日期和目标比赛集合比较。', '',
             '主比较固定为“直接分类树／表现＋全赛事 减 进球回归转胜平负／表现＋全赛事”的胜平负LL。'
             '全部逻辑回归、表现输入对照、提前量和赛季同时保留；没有按后季成绩选赢家或融合权重。', '',
             '| 算法 | 分析前固定设置（原计划记录） |', '|---|---|']
    lines[4:4] = primary_overview(result)
    for kind, config in plan['classifier_config'].items():
        lines.append(f'| {kind} | {config} |')
    docs = plan.get('sources', {})
    lines += ['', '多项逻辑回归直接学习三类概率，分类树将固定复杂度随机树的类概率聚合。'
              f'算法定义见[逻辑回归官方文档]({docs.get("logistic_docs", "")})与'
              f'[ExtraTrees官方文档]({docs.get("tree_docs", "")})；本轮实际设置以保存模型参数为准。', '',
              '分类树叶的至少30个样本指30场比赛；进球回归叶至少60个样本指计分方记录。'
              '这只粗略接近最低比赛规模，目标函数、损失与训练表示仍不同，不能把全部差值因果归于“改成直接胜平负”一个因素。'
              '两类方案都未搜索超参数网格。', '',
              '较早比赛采用180天半衰期权重，随后统一缩放使权重总和等于该次训练比赛数n。'
              '逻辑回归固定C＝1；本地实现的有效L2强度为1／(C×n)，因此随n变化，不能称为惩罚强度恒定。'
              '这与进球回归的惩罚不是等价设置，也没有根据本轮成绩再改C。树模型的全部权重乘同一正数不改变分裂目标的相对比较。', '']
    supplement = out / '权重与正则解释补充.md'
    if supplement.exists():
        lines += [f'原分析前计划的正则表述已在[权重与正则解释补充]({supplement.as_posix()})中澄清，原计划保留。', '']
    lines += ['缺失值由该次更早训练段中位数和缺失标记处理；逻辑回归另用训练段加权均值／标准差缩放。'
              '缺失列被保留时的内部填充值不代表真实观测或未知球队影响为零；未知基础球队仍明确列为不可预测。'
              '模型原生概率用于评分，不额外平滑、截断或重归一化；零类别概率按失败报告。', '',
              '## 时间、覆盖与实际拟合', '',
              '比赛日D、发布日F＝D减提前天数、参数日P满足P≤F＜D。参数日按既有28天周期，'
              '训练标签对应比赛日严格早于P，各历史训练例的输入也只用其自身发布日前的历史。'
              '没有随机拆分比赛行，也没有读取同场赛后统计。日历日期规则不等于精确24／72小时；'
              '历史资料首次发布、修订和改期公告时间仍有缺口。', '',
              f'起始历史赛季为{", ".join(plan["train_initial"])}；'
              f'较早比较为{plan["earlier_comparison"]}，后季为{plan["later_retrospective_comparison"]}。'
              '后季按既定周期继续仅用更早比赛更新；较早学习的进球／xG融合只在后季作参照。'
              '历史赛季此前已经查看，本轮是回溯研究，不是新的盲测。', '',
              '| 提前量 | 四季可用 | 明确不可预测 | 较早比较 | 后季 | 实际参数日期数 | 完成分类拟合 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for lead, item in result['horizons'].items():
        cover = coverage[lead]
        lines.append(f'| 前{lead}日 | {cover["n"]:,} | {len(cover["unavailable"])} | '
                     f'{item["validation"]["n"]:,} | {item["test"]["n"]:,} | '
                     f'{counts[lead]["cutoffs"]} | {counts[lead]["n"]} |')
    lines += ['', '| 提前量 | 每个分类方案实际拟合数 | 原始特征列数 |', '|---|---|---|']
    for lead, count in counts.items():
        fitted = '；'.join(f'{label(arm)}：{n}' for arm, n in count['arm_counts'].items())
        features = '；'.join(f'{family}：' + '/'.join(str(n) for n in sorted(values))
                            for family, values in count['feature_counts'].items())
        lines.append(f'| 前{lead}日 | {fitted} | {features} |')
    supports = [set(item['test']['match_ids']) for item in result['horizons'].values()]
    overlap = len(set.intersection(*supports))
    union = len(set.union(*supports))
    lines += ['', '不同提前量可用比赛略有差异，不能把两个总体均分直接相减当作提前日期效果；'
              '每个提前量内部严格比较相同比赛。前1日与前3日的目标高度重叠，不将两档场数相加当作独立比赛数量，'
              f'也不称两次独立重复实验；后季各档目标交集为{overlap:,}场、并集为{union:,}场。'
              '同一比赛可重复进入不同参数日期、算法和输入组的更早训练，'
              '审计中的训练引用条目数不是独立比赛数，拟合次数也不是独立验证次数。'
              '覆盖不足、原有异常和未知球队清单完整保留，没有为了分类线删除难例。', '']
    for lead, item in result['horizons'].items():
        validation, test = item['validation'], item['test']
        lines += [f'## 比赛前{lead}日：全部方案', '',
                  'LL、Brier和与RPS越低越好；Brier为三类平方误差之和，RPS按主胜、平局、客胜顺序计算。'
                  '方向命中是取最高概率类别后的三选一表现，不等于概率准确率、收益或中奖保证。', '',
                  '### 较早赛季', '']
        metric_table(lines, validation)
        lines += ['', '### 后季', '']
        metric_table(lines, test)
        contrasts = list(dict.fromkeys(key.rsplit(':', 1)[0] for key in test['difference_order']))
        main = [value for value in (PRIMARY, 'extra_trees_performance_minus_regression_performance')
                if value in contrasts]
        lines += ['', '### 预先指定主比较与相同表现输入对照', '',
                  '差值为左方案减右方案，负值代表损失下降。各区间来自同场差值的日历块重采样；'
                  '同时保留全部已保存块长，没有挑选最有利区间。', '']
        interval_table(lines, test, main)
        lines += ['', '### 逻辑回归、输入差异及较早融合对照', '']
        interval_table(lines, test, [value for value in contrasts if value not in main])
        lines += ['', '### 后季逐类别与分箱校准', '',
                  '平均概率与实际比例接近只说明总体边际接近，不说明每场概率或每个概率段可靠。'
                  '分箱加权绝对差逐类使用原来10个等宽概率段、按箱内比赛数加权；它是描述量，未校正小样本偏差，'
                  '也没有独立区间。实际该类LL用于观察错误概率是否集中在主胜、平局或客胜；不能只看整体平局均值。', '']
        calibration_table(lines, test, class_order)
        lines += ['', f'较早及后季的所有非空分箱、箱内场数和偏差见[逐类别概率分箱全表]({detail.as_posix()})。', '',
                  '### 五联赛主比较及逻辑回归对照', '',
                  '| 联赛 | 场数 | 分类树LL | 同输入进球回归LL | 树减回归LL | 树减回归Brier | 逻辑回归减回归LL |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for comp, section in item['by_league'].items():
            direct = section['scores']['extra_trees_allcomp']
            regression = section['scores']['regression_allcomp']
            logistic = section['scores']['logistic_allcomp']
            lines.append(f'| {LEAGUES.get(comp, comp)} | {section["n"]} | {direct["log_loss"]:.7f} | '
                         f'{regression["log_loss"]:.7f} | {signed(direct["log_loss"] - regression["log_loss"])} | '
                         f'{signed(direct["brier_sum"] - regression["brier_sum"])} | '
                         f'{signed(logistic["log_loss"] - regression["log_loss"])} |')
        lines += ['', '分联赛表是覆盖及异质性描述，未提供联赛级独立区间，不据此挑联赛上线或另调权重。', '']
    lines += ['## 当前结论及采用范围', '', final_conclusion, '', calibration_interpretation, '',
              '全部方案和输入组的结果均保留，较早或后季较好的单条结果不会自动成为正式方案。'
              '本轮没有按市场赔率计算价值，也没有检验统计＋LLM联合能否增益，不能据此声称优于市场或LLM。'
              '若未形成跨提前量、概率评分与校准的一致证据，保留失败或不确定结果，'
              '不通过继续扫描C、树深、叶大小或联赛权重来追逐本轮历史分数。', '',
              '三类概率只输出胜平负，不能唯一确定完整比分、总进球、半全场或让球分布；'
              '这些预测继续由原有相应线路负责。国内联赛回溯也不能直接证明欧战或跨联赛预测已成熟。', '']
    diagnostic = out / '较早分类机制诊断' / '较早分类误差机制诊断.md'
    if diagnostic.exists():
        lines += [f'较早赛季的[分类误差机制诊断]({diagnostic.as_posix()})单独保留：'
                  '它用于检查固定逻辑回归的过度确信、稀有输入与普通范围内的误差，'
                  '不把全部损失归于少数异常值，也不是先看后季再调整C或重训。'
                  '它不能证明其他逻辑回归设置或全部分类算法无效。', '']
    memo = out / '后续方向备忘.md'
    if memo.exists():
        lines += [f'[后续方向备忘]({memo.as_posix()})仅盘点未验证的核心缺口，'
                  '明确区分持续状态与已完成的年度创新／快照残差研究，以及国内联赛与原欧战用途的不同优先级。'
                  '本轮没有因分类结果自动启动新结构、训练或神经网络。', '']
    audit_table(lines, out, source)
    lines += ['', '日历块条件区间只反映在本次固定模型和设计下比赛时间聚集的波动，'
              '不涵盖历史研究选择、全部参数误差、资料首次发布与修订、来源遗漏等全部不确定性。'
              '分类树训练重要性也不是因果贡献，本报告未据此解释哪项信息导致预测成败。', '',
              f'[完整结果汇总]({(out / "直接胜平负完整汇总.json").as_posix()}) · '
              f'[分析前计划]({(out / "分析前计划.json").as_posix()}) · '
              f'[覆盖与不可预测清单]({(out / "覆盖与不可预测.json").as_posix()}) · '
              f'[逐类别概率分箱全表]({detail.as_posix()})', '']
    target = out / '直接胜平负实跑结论与后续.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    main()

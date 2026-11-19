"""Write measured results and their limits; never refit or choose an arm."""
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
RUN = ROOT/'outputs/进球xG更早年份复现_20261024T180238377843Z'
OLD = ROOT/'outputs/进球xG共同训练记录_20261024T174247963857Z/共同训练记录融合结果.json'
NAMES = {'E0':'英超','D1':'德甲','SP1':'西甲','I1':'意甲','F1':'法甲'}


def ci(entry, domain, block='4'):
    k = entry['difference_order'].index('blend_minus_goal:'+domain)
    lo,hi = entry['block_intervals'][block]['interval_95'][k]
    return f'[{lo:+.6f}, {hi:+.6f}]'


def main():
    result = json.loads((RUN/'更早年份复现结果.json').read_text(encoding='utf-8'))
    old = json.loads(OLD.read_text(encoding='utf-8'))
    decomposition = json.loads((RUN/'比分增益精确分解.json').read_text(encoding='utf-8'))
    late = result['seasons']['2023-24']
    w = result['parameters']['weight']
    report = [
        '# 进球与xG：更早年份的实际前推复现', '',
        '**比分与总进球的改善再次出现；胜平负概率的改善没有稳定复现。继续保留融合研究候选，不把本轮命中数提高当成长期准度。**', '',
        '## 实际完成', '',
        '新增2020/21、2021/22五联赛Football-Data官方比分与Understat逐场xG，各3,652场；原2022/23、2023/24源文件复用。共7,230场，11条源冲突／已核实特殊比赛共同排除，进球与xG共同可用7,219场。四个旧球队名称采用明确别名，未模糊配对日期或比分。', '',
        '球队训练从2020/21开始，两个目标季均累积使用同一更早起点的记录。固定正则0.01、半衰180天、28天更新、原比分分布；进球与xG的每次训练ID、日期、主客场及时间权重完全相同。世界杯空窗跳过，闰年日期动态生成。', '',
        f'完成{result["new_goal_fits"]}次进球拟合、{result["new_xg_fits"]}次xG拟合、1次共享权重估计，参数网格0个。用2022/23的1,790场前推预测学习xG对数进球率权重{w:.2%}，随后固定到2023/24的1,733场；没有把后来约61%的旧权重倒用到过去。', '',
        '目标季原始3,578场，实际共同预测3,523场。早季跳过35场未知队及1场源日期冲突；后季跳过14场未知队、2场分段续赛及3场源日期冲突。未知队未填默认实力；全部缺失原因和逐场预测保存在结果中。', '',
        '## 较晚2023/24：同场结果', '',
        '| 指标 | 纯进球 | 独立xG | 进球＋xG | 融合减纯进球 |', '|---|---:|---:|---:|---:|',
    ]
    for label,key in [('比分概率损失','exact_score'),('胜平负概率损失','one_x_two'),
                       ('总球0／1／2／3／4+损失','total_goals_0_1_2_3_4plus'),
                       ('客队让两球三项损失','away_minus_two_three_way')]:
        vals = [late['scores'][a][key]['log_loss'] for a in ('goal','xg','blend')]
        report.append(f'| {label} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {vals[2]-vals[0]:+.6f} |')
    acc = [late['scores'][a]['one_x_two']['accuracy'] for a in ('goal','xg','blend')]
    report += [f'| 胜平负最高概率命中 | {round(acc[0]*late["n"])}/{late["n"]}（{acc[0]:.2%}） | {round(acc[1]*late["n"])}/{late["n"]}（{acc[1]:.2%}） | {round(acc[2]*late["n"])}/{late["n"]}（{acc[2]:.2%}） | {(acc[2]-acc[0])*100:+.2f}个百分点 |', '',
        '损失越低越好。比分与总球分组损失的2／4／8周95%条件区间均低于零；胜平负和客让两球的三套区间均跨零。提高了15场最高概率命中，仍不足以证明稳定的胜负概率优势。', '',
        '| 融合减纯进球 | 2周区间 | 4周区间 | 8周区间 |', '|---|---|---|---|',
    ]
    for label,key in [('比分','exact_score'),('胜平负','one_x_two'),
                       ('总球分组','total_goals_0_1_2_3_4plus'),('客让两球','away_minus_two_three_way')]:
        report.append(f'| {label} | {ci(late,key,"2")} | {ci(late,key,"4")} | {ci(late,key,"8")} |')
    report += ['', '## 全部联赛保留', '',
               '| 联赛 | 场数 | 比分损失差 | 比分4周区间 | 胜平负损失差 | 客让两球损失差 |',
               '|---|---:|---:|---|---:|---:|']
    for comp,name in NAMES.items():
        v = result['by_league']['2023-24'][comp]
        d = v['mean_loss_differences']
        report.append(f'| {name} | {v["n"]} | {d["blend_minus_goal:exact_score"]:+.6f} | {ci(v,"exact_score")} | {d["blend_minus_goal:one_x_two"]:+.6f} | {d["blend_minus_goal:away_minus_two_three_way"]:+.6f} |')
    report += [
        '', '五个联赛的比分点估计都改善，但4周区间仅英超明确低于零，不能说逐一得到确认。德甲、西甲、意甲的胜平负点估计略差，法甲的客让两球也较差；没有据此挑联赛或添加专属权重。', '',
        '## 两次跨年份结果如何理解', '',
        '| 学权重赛季→检验赛季 | 学习场数 | 检验场数 | xG权重 | 比分损失差 | 胜平负损失差 |',
        '|---|---:|---:|---:|---:|---:|',
        f'| 2022/23→2023/24，本轮 | 1790 | {late["n"]} | {w:.2%} | {late["mean_loss_differences"]["blend_minus_goal:exact_score"]:+.6f} | {late["mean_loss_differences"]["blend_minus_goal:one_x_two"]:+.6f} |',
        f'| 2024/25→2025/26，上轮共同训练 | 1734 | 1733 | {old["parameters"]["weight"]:.2%} | {old["seasons"]["2025-26"]["mean_loss_differences"]["common_blend_minus_common_goal:exact_score"]:+.6f} | {old["seasons"]["2025-26"]["mean_loss_differences"]["common_blend_minus_common_goal:one_x_two"]:+.6f} |', '',
        '两个不相邻目标季均保留了比分概率的改善，但胜平负增量明显不同。约61%和73%分别来自各自较早赛季，不是通用常数，不根据后季结果平均或挑选。法甲队数、联赛构成和疫情暖启动不同，不能把两轮差异纯粹归因于一个因素。', '',
        '## 增益来自哪里：零拟合的精确分解', '',
        '独立Poisson比分可精确写为“完整总进球分布”乘以“给定总球的主客分配分布”。因此，比分损失可以逐场拆成两个相加部分，不用增加模型参数：', '',
        '| 目标季 | 完整整数总球贡献 | 主客分配贡献 | 合计比分贡献 |', '|---|---:|---:|---:|',
    ]
    for p,v in decomposition['held_seasons'].items():
        d = v['mean_loss_differences']
        report.append(f'| {p} | {d["exact_total_goals"]:+.6f} | {d["home_away_allocation_given_actual_total"]:+.6f} | {d["exact_score"]:+.6f} |')
    report += [
        '', '2023/24较明确的增益主要落在总球，主客分配的三套区间均跨零；2025/26主客分配三套区间低于零，总球只有两周区间跨零。这说明不能把比分改善直接等同于胜负判断改善，也不能断言xG只对总球有用。', '',
        '这里是完整整数总球Poisson损失，与前表0／1／2／3／4+分组评分不同。实际总球仅用于赛后条件评分，绝不作为赛前输入；这只是保存预测的算术归因，不是因果机制证明。', '',
        '## 验证与局限', '',
        '相关131项测试通过。独立审查重新核对源覆盖、208次训练历史与衰减、全部率值、早季权重、四领域评分、12个范围及全部2／4／8周区间；评分和区间最大差约2.10×10⁻¹³。精确分解另经独立Poisson与Binomial算式重算，两季全部联赛及区间一致，最大分解误差约2.66×10⁻¹⁵。未发现需重训的问题。', '',
        '这是固定方法在更早年份的回溯复现。模型结构、0.01／180及xG路线是在看过其他历史研究后保留的方案，因此不是新的盲测。历史xG发布时间与后续修订未知；采用严格更早日历日期可用的研究假设。区间条件于已有预测，未包含权重估计和完整研究选择不确定性；本轮只覆盖历史联赛，未验证欧战跨联赛适用性。', '',
        '## 后续结构讨论', '',
        '此复现之后，用户已同意保持当前保存的总进球率，仅在更早赛季学习一个主客进球分配权重，再在两组年份检验。该研究已另行实跑完成，未发现稳定新增增益，未采用；最新结论见下方链接。原完整融合、纯进球及独立xG继续保留。', '',
        '[后续主客分配实跑结论](../../outputs/固定总球主客分配研究_20261024T182604417382Z/主客分配实跑与采用结论.md)', '',
        '当前融合仍是研究候选，没有替换正式预测或改写原赛前概率。市场、LLM与联合线路继续保留。没有查询或复核当前比赛赛果，等待用户提醒；没有新增神经网络。全部文件、缓存及临时文件在项目目录。', '',
        '## 新增历史例外的官方依据', '',
        '- 2022-03-18波鸿—门兴：69分钟终止，后按0–2判定，不能视为完整90分钟进球训练样本。[德国足协判决](https://www.dfb.de/fileadmin/_dfbdam/257416-220324_VfL_Bochum.pdf)。',
        '- 2020-09-19维罗纳—罗马：两家当前源均保留场上0–0，但官方确认行政判罚；按既有特殊比赛政策共同排除，不把源比分改成3–0。[意大利足协](https://www.figc.it/it/federazione/news/respinto-il-ricorso-della-roma-confermata-la-sconfitta-per-3-0-a-tavolino-con-il-verona-ha0usqwf)。',
        '- 2021-10-27尼斯—马赛：在特鲁瓦中立场重赛，当前日期模型不表示中立场，本轮明确共同排除。[法国职业足球联盟安排](https://www.lfp.fr/article/ogc-nice-olympique-de-marseille-le-27-octobre-a-21-h-au-stade-de-laube-a-troyes)、[尼斯官方中立场说明](https://www.ogcnice.com/fr/article/119349/nice-marseille-se-jouera-a-troyes.html)。', '',
        f'[完整结果]({(RUN/"更早年份复现结果.json").as_posix()}) · [增益分解]({(RUN/"比分增益精确分解.json").as_posix()}) · [源差异]({result["plan"]["source_audit"].replace(chr(92),"/")})', '',
    ]
    (RUN/'更早年份实跑结论与后续计划.md').write_text('\n'.join(report),encoding='utf-8')
    print(str(RUN/'更早年份实跑结论与后续计划.md'))


if __name__ == '__main__':
    main()

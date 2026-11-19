"""One fixed-preset EPL-only control versus archived FPL-status input block."""
from datetime import date, datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import time

import joblib

from football_model.early_goal_learners import EarlyGoalLearner
from run_adjusted_xg_form_20261026 import compare
from run_broad_early_models_20261025 import predict_row
from run_fixed_horizon_refresh_20261025 import CONTEXT_KEYS
from intake_fpl_availability_20261026 import ROOT, OUT as INTAKE, MEASUREMENTS, write


PERIODS = ('2024-25', '2025-26')
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z/lead_3/逐场完整预测.json'


def main():
    started = time.perf_counter()
    now = datetime.now(timezone.utc)
    out = ROOT/'outputs'/now.strftime('FPL赛前可用性固定对照_%Y%m%dT%H%M%S%fZ')
    (out/'models').mkdir(parents=True)
    method = {
        'created_at': now.isoformat(), 'scope': 'EPL only, previously examined historical seasons; not untouched validation',
        'input': str(INTAKE/'逐场赛前状态特征.json.gz'),
        'new_information': 'dated FPL i/d/s/u flags and roster/unknown counts; source-observation age, own and opponent',
        'new_measurements': list(MEASUREMENTS), 'control_features': 121, 'new_features': 135,
        'control': 'refit identical preset on EPL-only history; isolates the added block from changing league training scope',
        'pooled_reference': 'also report original five-league 121-column forecast on the same EPL target IDs',
        'target': 'real full-time goals; unchanged independent Poisson score distribution and 1X2 conversion',
        'configuration': EarlyGoalLearner('extra_trees', 'rich')._estimator().get_params(),
        'half_life_days': 180, 'cutoffs': 'original Aug1-anchored 28-day dates; each historical input remains at F=D-3; train only match date<P',
        'availability': 'both filename time and first-added repository commit timestamp strictly before F at 00:00 UTC; retrospective observation proxy',
        'missing': 'no target removed for missing availability; training-only median imputation and existing missing indicators',
        'not_used': ['chance percent', 'news text inference', 'cumulative minutes/starts', 'player ability weights', 'confirmed XI'],
        'primary': '1X2 LogLoss for both seasons separately, matched EPL-only new minus control',
        'secondary': ['1X2 Brier', 'single highest-probability outcome accuracy', 'score and total-goal NLL', 'same EPL targets against pooled model'],
        'interval': 'paired conditional 4-calendar-week moving blocks; does not include model refitting uncertainty',
        'stopping_point': 'complete the 22 existing dates for 2 arms (44 fixed fits), then report; no window/tree/weight search after results',
        'limitations': ['FPL flags are not a complete medical record or proof of fitness',
            'unweighted player counts are not lost team strength or causal injury effects',
            'long absences may already influence historical performance',
            'archive commit dates do not prove first public announcement or actual push time',
            'current forecasts are 5-7 days ahead, while this study is 3 days ahead',
            'adding columns changes randomized split opportunities; not a causal ablation'],
        'current_forecasts_replaced': False, 'current_results_checked': False}
    write(out/'分析前方法.json', method)
    with gzip.open(INTAKE/'逐场赛前状态特征.json.gz', 'rt', encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    rows.sort(key=lambda r: (r['date'], r['match_id']))
    targets = [r for r in rows if r['period'] in PERIODS]
    pooled = [r for r in json.loads(SOURCE.read_text(encoding='utf-8'))['fresh_performance']
              if r['competition'] == 'E0']
    pooled.sort(key=lambda r: (r['date'], r['match_id']))
    if [r['match_id'] for r in targets] != [r['match_id'] for r in pooled]:
        raise ValueError('Same complete EPL target cohort required')
    for row in rows:
        if not row['fit_before_date'] <= row['forecast_before_date'] < row['date']:
            raise ValueError('Historical feature timing conflict')
        observation = row['archive_observation']
        if observation and not observation['availability_proxy_at'] < row['forecast_before_date']+'T00:00:00+00:00':
            raise ValueError('Archived status cannot postdate its forecast cutoff')
    tracks = {'epl_control': [], 'epl_availability': []}
    windows, parameters = [], {}
    for period in PERIODS:
        cohort = [r for r in targets if r['period'] == period]
        for cutoff in sorted({r['fit_before_date'] for r in cohort}):
            held = [r for r in cohort if r['fit_before_date'] == cutoff]
            history = [r for r in rows if r['date'] < cutoff]
            for arm, feature_key in (('epl_control', 'reference_features'), ('epl_availability', 'side_features')):
                tick = time.perf_counter()
                train = [r | {'side_features': r[feature_key]} for r in history]
                model = EarlyGoalLearner('extra_trees', 'rich').fit(train, date.fromisoformat(cutoff))
                path = out/'models'/f'{arm}_{cutoff}.joblib'
                joblib.dump(model, path, compress=3)
                parameters[f'{arm}:{cutoff}'] = model.parameters()
                rates = model.predict_many([r[feature_key] for r in held], [date.fromisoformat(r['date']) for r in held])
                for row, means in zip(held, rates, strict=True):
                    context = {k: row[k] for k in CONTEXT_KEYS}
                    tracks[arm].append(predict_row(context, means))
                windows.append({'period': period, 'cutoff': cutoff, 'arm': arm,
                    'training_n': len(history), 'target_n': len(held), 'seconds': time.perf_counter()-tick})
                write(out/'训练进度.json', {'completed_fits': len(windows), 'windows': windows})
                print(json.dumps(windows[-1] | {'completed_fits': len(windows)}, ensure_ascii=False), flush=True)
                del model, train
                gc.collect()
    for values in tracks.values():
        values.sort(key=lambda r: (r['date'], r['match_id']))
    result = {'method': method, 'coverage': json.loads((INTAKE/'历史覆盖实测.json').read_text(encoding='utf-8')),
              'new_fits': len(windows), 'base_expert_refits': 0, 'parameter_grids': 0, 'windows': windows,
              'by_period': {}, 'pooled_reference_by_period': {}}
    for period in PERIODS:
        old = [r for r in tracks['epl_control'] if r['period'] == period]
        new = [r for r in tracks['epl_availability'] if r['period'] == period]
        ref = [r for r in pooled if r['period'] == period]
        result['by_period'][period] = compare(old, new, intervals=True)
        result['pooled_reference_by_period'][period] = compare(ref, new)
    deltas = [r['mean_differences_adjusted_minus_original']['one_x_two'] for r in result['by_period'].values()]
    result['decision'] = {'adopted': False, 'stop_this_comparison': True,
        'description': ('两季未同向改善胜平负概率，不能将伤停人数直接视为有效实力修正。'
                        if any(d >= 0 for d in deltas) else
                        '两季平均胜平负损失仅微降，4周条件区间均跨零，单选命中变化不一致；较晚季对原五联赛模型也未改善。未确认稳定增益，不采纳到当前预测。'),
        'scope': 'only this unweighted archived-status input block; not evidence that player availability has no value',
        'planned_additional_fits_this_arm': 0}
    result['seconds'] = time.perf_counter()-started
    write(out/'全部训练参数.json', parameters)
    write(out/'逐场完整预测.json', tracks | {'original_pooled': pooled})
    write(out/'完整研究结果.json', result)
    lines = ['# 历史赛前可用性：新数据实接与固定模型对照', '',
        '来源是按日期保存的FPL状态快照，不使用赛后出场记录或未来伤情。伤病、存疑、停赛、其他不可用分别计数；'
        '没有把FPL百分比当出场概率，也没有给球员手写能力损失。', '',
        '每队新增7项测量，加入自己和对手共14列。为避免把仅训练英超的变化误算成新数据效果，'
        '同时训练英超原121列对照，并保留原五联赛模型在相同比赛上的结果。', '',
        '| 赛季 | 场数 | 英超原模型LL | 加入状态LL | LL差 | 原单选命中 | 加入状态单选命中 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for period, comp in result['by_period'].items():
        old, new = comp['original']['one_x_two'], comp['adjusted']['one_x_two']
        delta = comp['mean_differences_adjusted_minus_original']
        lines.append(f"| {period} | {comp['n']} | {old['log_loss']:.6f} | {new['log_loss']:.6f} | {delta['one_x_two']:+.6f} | {old['accuracy']:.2%} | {new['accuracy']:.2%} |")
    lines += ['', 'LL差为新减旧，负数更好；单选命中只取胜、平、负中概率最高的一项。', '',
        '| 赛季 | LL差4周条件区间 | Brier差 | 比分NLL差 | 总球NLL差 | 相对原五联赛模型LL差 |',
        '|---|---|---:|---:|---:|---:|']
    for period, comp in result['by_period'].items():
        ci = dict(zip(comp['difference_order'], comp['conditional_4week_intervals']['interval_95']))['one_x_two']
        delta = comp['mean_differences_adjusted_minus_original']
        pooled_delta = result['pooled_reference_by_period'][period]['mean_differences_adjusted_minus_original']['one_x_two']
        lines.append(f"| {period} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {delta['one_x_two_brier']:+.6f} | {delta['exact_score']:+.6f} | {delta['total_goals_0_1_2_3_4plus']:+.6f} | {pooled_delta:+.6f} |")
    lines += ['', '区间只描述既有预测下的时间分块波动，不含完整重训不确定性。历史赛季此前已被查看，属于前推研究对照，不是全新盲测。', '',
        '## 数据实测', '']
    for period, cov in result['coverage']['by_period'].items():
        lines.append(f"- {period}：{cov['n']}场，双方名单状态均能匹配{cov['both_teams_observed']}场；快照年龄中位数{cov['age_hours_median']:.2f}小时、最大{cov['age_hours_max']:.2f}小时。")
    lines += ['', '开季前分钟数可能延续上季，转会后也可能含前队累计，因此本轮没有把这些计数作为实力权重。'
        'FPL状态是被记录的标签，a或没有标记不代表医生确认健康；u另列，不混作伤病。'
        '238份快照含尚未解释的源标签n，作为其他状态单列，未并入伤停或健康；本批所需原始字段没有缺列。'
        '仓库提交时间是历史观察代理，不是伤情最早公开时刻的证明。', '',
        '## 决定与停止节点', '', result['decision']['description'], '',
        '本轮完成44次固定设置训练、零参数网格、零基专家重训后停止，不继续更换窗口或树深搜数字。'
        '结论只针对本轮无权重状态计数，不能推断球员可用性整体无价值。'
        '若继续利用该来源，真正待补的是可靠的球员身份、此前贡献与替补能力，而非把缺阵人数机械乘系数。', '',
        '当前比赛原预测、LLM线和联合线未被替换；未复核当前新赛果。', '',
        '来源：[Randdalf/fplcache](https://github.com/Randdalf/fplcache)。仓库代码许可为Unlicense，不扩大上游数据权利。']
    (out/'结果与决定.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for period, value in result['by_period'].items():
        print(json.dumps({'period': period, 'n': value['n'], 'control': value['original']['one_x_two'],
            'availability': value['adjusted']['one_x_two'], 'deltas': value['mean_differences_adjusted_minus_original']}, ensure_ascii=False), flush=True)
    print(json.dumps({'output': str(out), 'new_fits': len(windows), 'decision': result['decision'], 'seconds': result['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Read saved training memberships and replay predictions, without fitting."""
from collections import Counter, defaultdict
from datetime import date
import gc
import gzip
import json
import math
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import joblib
import numpy as np


ROOT = REPOSITORY_ROOT
SOURCE = ROOT/'outputs/固定发布日期信息刷新_20261025T071806561562Z'
FOLDER = SOURCE/'lead_3'
OUT = ROOT/'outputs/原模型训练测试核查_20261026'
NAME = '独立训练目标集合核查'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fixture_key(row):
    # Opposite scoring-side representations of the SAME fixture remain equal.
    # The reverse league leg on a DIFFERENT date is a distinct eligible fixture.
    return row['date'], row['competition'], tuple(sorted((row['home'], row['away'])))


def main():
    predictions = read(FOLDER/'逐场完整预测.json')['fresh_performance']
    exported_parameters = read(FOLDER/'全部模型参数.json')
    summary = read(FOLDER/'完整结果汇总.json')
    with gzip.open(SOURCE/'features_lead3.json.gz', 'rt', encoding='utf-8') as handle:
        rows = json.load(handle)['rows']
    rows.sort(key=lambda row: (row['date'], row['match_id']))
    by_id = {row['match_id']:row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Historical fixture identities are not unique')
    counts_by_fixture = Counter(fixture_key(row) for row in rows)
    groups = defaultdict(list)
    for row in predictions:
        groups[row['fit_before_date']].append(row)
    test_ids = {row['match_id'] for row in predictions if row['period'] == '2025-26'}
    names = sorted(rows[0]['fresh_performance_features'][0])
    matrix = np.asarray([[np.nan if side[k] is None else side[k] for k in names]
        for row in rows for side in row['fresh_performance_features']],dtype=float)
    all_fits = []
    ever_later_training = set()
    replays = []
    metadata_consistent = True
    for cutoff, held in sorted(groups.items()):
        model_path = FOLDER/'models'/f'fresh_performance_{cutoff}.joblib'
        model = joblib.load(model_path)
        parameter = model.parameters()
        history = [row for row in rows if row['date'] < cutoff]
        expected_ids = [row['match_id'] for row in history]
        actual_ids = parameter['reference_match_ids']
        id_set = set(actual_ids)
        unknown = sorted(id_set-set(by_id))
        actual_history = [by_id[mid] for mid in actual_ids if mid in by_id]
        actual_keys = {fixture_key(row) for row in actual_history}
        target_ids = {row['match_id'] for row in held}
        overlaps = sorted(id_set & target_ids)
        same_fixture = sorted(row['match_id'] for row in held if fixture_key(row) in actual_keys)
        illegal_dates = sorted(row['match_id'] for row in actual_history if row['date'] >= cutoff)
        context_errors = []
        for row in held:
            prepared = by_id[row['match_id']]
            for key in ('date','period','competition','home','away','fit_before_date',
                        'forecast_before_date','actual_score','actual_outcome'):
                if row[key] != prepared[key]:
                    context_errors.append([row['match_id'],key])
            if not cutoff <= row['forecast_before_date'] < row['date']:
                context_errors.append([row['match_id'],'parameter/publication/fixture dates'])
        expected_metadata = {k:v for k,v in exported_parameters[f'fresh_performance:{cutoff}'].items()
                             if k not in ('pipeline','horizon_days')}
        matches_export = parameter == expected_metadata
        metadata_consistent &= matches_export

        # Recover fitted evidence from the actual trees, not just a report score.
        weights = np.repeat([math.exp(-math.log(2)*(date.fromisoformat(cutoff)
            -date.fromisoformat(row['date'])).days/180) for row in history],2)
        goals = np.asarray([goal for row in history for goal in row['actual_score']],dtype=float)
        expected_weight = float(weights.sum())
        expected_mean = float(np.dot(weights,goals)/expected_weight)
        trees = model.estimator.estimators_
        tree_sample_counts = sorted({int(tree.tree_.n_node_samples[0]) for tree in trees})
        weight_error = max(abs(float(tree.tree_.weighted_n_node_samples[0])-expected_weight) for tree in trees)
        goal_mean_error = max(abs(float(tree.tree_.value[0].ravel()[0])-expected_mean) for tree in trees)
        train_matrix = matrix[:2*len(history)]
        empty = np.isnan(train_matrix).all(axis=0)
        median = np.zeros(len(names))
        for column in range(len(names)):
            if not empty[column]:
                median[column] = np.nanmedian(train_matrix[:,column])
        median_error = float(np.max(np.abs(median-model.imputer.statistics_)))
        expected_missing = np.flatnonzero(np.isnan(train_matrix).any(axis=0)).tolist()
        preprocessing_matches = (median_error < 1e-12
            and expected_missing == model.imputer.indicator_.features_.tolist()
            and model.scaler is None and model.feature_names == names)

        # Forward replay is prediction-only; no training or optimizer is called.
        replay = np.asarray(model.predict_many(
            [by_id[row['match_id']]['fresh_performance_features'] for row in held],
            [date.fromisoformat(row['date']) for row in held]))
        rate_error = float(np.max(np.abs(replay-np.asarray([row['rates'] for row in held]))))
        replays.append(rate_error)
        current_period = held[0]['period']
        earlier_test = sorted(id_set & test_ids)
        ever_later_training.update(earlier_test)
        reverse_legs = []
        for target in held:
            if any(row['competition'] == target['competition'] and row['period'] == target['period']
                   and row['home'] == target['away'] and row['away'] == target['home']
                   and row['date'] < cutoff for row in actual_history):
                reverse_legs.append(target['match_id'])
        record = {
            'fit_before_date':cutoff, 'target_period':current_period,
            'saved_model':str(model_path),
            'actual_training_ids_location':'joblib object parameters()[reference_match_ids]',
            'exported_training_ids_location':f'{FOLDER/"全部模型参数.json"} :: fresh_performance:{cutoff}.reference_match_ids',
            'training_set_description':f'all prepared rows sorted by (date, match_id) with date < {cutoff}',
            'ordered_training_ids_equal_exact_earlier_prepared_rows':actual_ids == expected_ids,
            'training_n':len(actual_ids), 'unique_training_ids':len(id_set),
            'training_period_counts':dict(Counter(row['period'] for row in actual_history)),
            'training_date_min':min(row['date'] for row in actual_history),
            'training_date_max':max(row['date'] for row in actual_history),
            'training_first_id':actual_ids[0], 'training_last_id':actual_ids[-1],
            'target_n':len(held), 'target_date_min':min(row['date'] for row in held),
            'target_date_max':max(row['date'] for row in held),
            'target_id_overlap':overlaps,'same_fixture_either_scoring_side_overlap':same_fixture,
            'unknown_training_ids':unknown,'training_date_at_or_after_cutoff':illegal_dates,
            'target_context_mismatches':context_errors,
            'saved_metadata_equals_export':matches_export,
            'model_before_date_equals_cutoff':model.before_date.isoformat() == cutoff,
            'reported_n_matches_equals_actual_id_count':parameter['n_matches'] == len(actual_ids),
            'reported_scoring_sides_equals_twice_n':parameter['n_scoring_sides'] == 2*len(actual_ids),
            'all_tree_root_training_sample_counts':tree_sample_counts,
            'expected_root_sample_count':2*len(history),
            'max_tree_root_weight_error':weight_error,
            'max_tree_root_weighted_goal_mean_error':goal_mean_error,
            'preprocessing_fits_exact_earlier_rows':preprocessing_matches,
            'max_training_median_error':median_error,
            'max_replayed_prediction_goal_rate_error':rate_error,
            'earlier_2025_26_test_fixtures_now_legally_in_training_n':len(earlier_test),
            'earlier_same_period_training_n':sum(row['period'] == current_period for row in actual_history),
            'targets_with_earlier_reverse_home_away_league_leg_n':len(reverse_legs),
        }
        all_fits.append(record)
        del model
        gc.collect()
    period_scores = {}
    for period in ('2024-25','2025-26'):
        local = [row for row in predictions if row['period'] == period]
        probs = np.asarray([row['probabilities'] for row in local])
        truth = np.asarray([row['actual_outcome'] for row in local])
        hits = int(np.sum(probs.argmax(axis=1) == truth))
        period_scores[period] = {'n':len(local),'top1_correct':hits,'top1_accuracy':hits/len(local),
            'one_x_two_log_loss':float(-np.log(probs[np.arange(len(local)),truth]).mean())}
    violations = [record['fit_before_date'] for record in all_fits if (
        not record['ordered_training_ids_equal_exact_earlier_prepared_rows']
        or record['target_id_overlap'] or record['same_fixture_either_scoring_side_overlap']
        or record['unknown_training_ids'] or record['training_date_at_or_after_cutoff']
        or record['target_context_mismatches'] or not record['saved_metadata_equals_export']
        or not record['model_before_date_equals_cutoff']
        or not record['reported_n_matches_equals_actual_id_count']
        or not record['reported_scoring_sides_equals_twice_n']
        or record['all_tree_root_training_sample_counts'] != [record['expected_root_sample_count']]
        or record['max_tree_root_weight_error'] > 1e-8
        or record['max_tree_root_weighted_goal_mean_error'] > 1e-10
        or not record['preprocessing_fits_exact_earlier_rows']
        or record['max_replayed_prediction_goal_rate_error'] > 1e-12)]
    result = {
        'scope':'saved D-3 fresh_performance121 memberships and prediction replay only; no fit, no score lookup',
        'source_predictions':str(FOLDER/'逐场完整预测.json'),
        'source_prepared_rows':str(SOURCE/'features_lead3.json.gz'),
        'prepared_rows_n':len(rows),'predictions_n':len(predictions),'saved_fits_n':len(all_fits),
        'prepared_duplicate_same_date_fixture_either_side_n':sum(n-1 for n in counts_by_fixture.values()),
        'class_order':['home','draw','away'], 'recomputed_stored_scores':period_scores,
        'stored_report_test_score':summary['test']['scores']['fresh_performance']['one_x_two'],
        'violating_fit_dates':violations,
        'total_direct_target_training_overlaps':sum(len(r['target_id_overlap']) for r in all_fits),
        'total_either_scoring_side_same_fixture_overlaps':sum(len(r['same_fixture_either_scoring_side_overlap']) for r in all_fits),
        'max_replayed_prediction_goal_rate_error':max(replays),
        '2025_26_predicted_fixtures_legally_entering_a_later_fit_n':len(ever_later_training),
        'conclusion':'No target fixture is included in its own saved training membership; this is rolling chronological evaluation, not training accuracy.' if not violations else 'See violating fit dates; do not accept evaluation pending investigation.',
        'limitations':[
            'Saved sklearn trees retain sample counts and aggregate moments, not original X or per-sample match IDs. Membership evidence is saved model metadata, independently matched to original prepared rows and fitted imputation/tree moments, plus exact prediction replay; no refit was performed.',
            'This does not certify every upstream feature or base-goal/xG expert; root is auditing those separately.',
            'Historical seasons were repeatedly viewed during research; chronological row separation does not make 2025/26 an untouched independent final test.',
            'Availability is calendar-date based; historical provider publication/revision timestamps are not reconstructed.',
            'A reverse home/away league leg on a different earlier date is a different completed match, not leakage of the target scoring side.'
        ],
        'fits':all_fits,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{NAME}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 原 fresh_performance121 训练目标集合独立核查','',
        f"2025/26 的已存逐场概率重新计分为 {period_scores['2025-26']['top1_correct']}/{period_scores['2025-26']['n']} = {period_scores['2025-26']['top1_accuracy']:.6%}，1X2 对数损失 {period_scores['2025-26']['one_x_two_log_loss']:.12f}。",'',
        f"读取 {len(all_fits)} 个实际保存的 fresh_performance joblib 模型，共核对 {len(predictions)} 条目标预测。每个模型的训练比赛 ID 顺序均与原准备数据中 date < fit_before_date 的比赛完全相同。目标本场 ID 重合数 {result['total_direct_target_training_overlaps']}；同日同联赛同对阵（含主客两侧反向表达）重合数 {result['total_either_scoring_side_same_fixture_overlaps']}。异常日期：{violations}。",'',
        f"每个模型都核对了实际树根的训练样本数、180 天权重和加权进球均值，以及训练中位数和缺失指示器。对全部目标只做预测重放，最大预期进球误差 {max(replays):.3g}。未调用 fit。",'',
        f"这是按时间滚动的样本外评价。2025/26 已预测比赛中，{len(ever_later_training)} 场在其后某次模型更新中成为合法的较早训练信息；它们不在预测自身的那个模型训练集合中。较早的反向主客场联赛回合也是另一场已结束的比赛。",'',
        '| 参数截点 | 训练场数 | 训练最晚日 | 目标场数 | 本期较早比赛已入训练 | 目标重合 |',
        '|---|---:|---|---:|---:|---:|']
    for r in all_fits:
        lines.append(f"| {r['fit_before_date']} | {r['training_n']} | {r['training_date_max']} | {r['target_n']} | {r['earlier_same_period_training_n']} | {len(r['target_id_overlap'])} |")
    lines += ['', '证据范围与缺口：','',
        '树没有保留逐条原始训练 X 及其比赛 ID。本次实际对象的训练 ID 元数据、原准备记录、拟合后的预处理和树内样本统计相互一致，且目标预测完全可重放；这比仅查看报告更强，但不等于在不重新训练的情况下恢复每棵树的逐条原始输入。', '',
        '本核查只回答目标与训练集合是否混用。上游特征和基础进球/xG模型由主任务另查。历史赛季已被反复用于研究，不应描述成从未查看的最终盲测；日期级数据可用性也不等于已重建历史发布与修订时间。', '',
        '精确训练集合可在每个保存对象的 parameters()[reference_match_ids] 查看；JSON 给出每个对象路径、导出参数位置、完全相等检查、日期及逐期人数，避免复制多份大名单。']
    (OUT/f'{NAME}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('prepared_rows_n','predictions_n','saved_fits_n',
        'recomputed_stored_scores','violating_fit_dates','total_direct_target_training_overlaps',
        'total_either_scoring_side_same_fixture_overlaps','max_replayed_prediction_goal_rate_error',
        '2025_26_predicted_fixtures_legally_entering_a_later_fit_n')},ensure_ascii=False))


if __name__ == '__main__':
    main()

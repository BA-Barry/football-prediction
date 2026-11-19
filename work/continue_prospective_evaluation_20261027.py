"""Connect saved current forecasts to the existing scorer, without fitting or I/O services.

Each source version remains a separate cohort. No result-based version choice,
probability regeneration, timestamp refresh, pooling, or network access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

from football_model.domain import Fixture
from football_model.multiline import TRACKS, _summary, multi_record, score_multi
from football_model.prospective import _time

ROOT = REPOSITORY_ROOT
CURRENT = Path('outputs/当前丰富信息预测_20261026T030805654404Z')
SOURCES = {
    'original10': 'outputs/赛果对比_20261026/原10场_原预测与赛果.json',
    'personnel_revision9': 'outputs/赛果对比_20261026/球员资料修订9场_原预测与赛果.json',
    'next_original10': 'data/prematch-intelligence/20261024_batch/run_20261024T150900467071Z/batch_case.json',
    'next_context_revision7': 'data/prematch-intelligence/20261024_batch/revision_20261024T154941678960Z/batch_case.json',
}
LABELS = {'original10': '原轮首次10场', 'personnel_revision9': '原轮人员资料修订9场',
          'next_original10': '下一轮10月24日原版10场',
          'next_context_revision7': '下一轮资料修订7场',
          'next_simple10': '下一轮10月26日简单模型10场',
          'next_rich10': '下一轮10月26日丰富模型10场',
          'next_allcomp_repaired10': '下一轮全赛事资料修复10场',
          'next_planned_cup_llm10': '下一轮计划杯赛信息独立LLM10场',
          'next_availability_llm10': '下一轮官方可用性独立LLM10场'}
ARM_LABELS = {'statistical': '统计模型', 'llm_independent': '独立LLM', 'market': '市场缓存',
              'llm_model_informed': '模型知情LLM', 'joint_pool': '原50:50联合'}
BJ = timezone(timedelta(hours=8))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                          encoding='utf-8')


def current_cohorts(root):
    """Adapt actual archived generation times; the data cutoff is not publication."""
    folder = root / CURRENT
    stats = read(folder / '完整预测与来源.json')
    comparison = read(folder / '多线逐场对照.json')
    llm = read(folder / '本次独立LLM预测.json')
    statistics = {r['fixture']['match_id']: r for r in stats['records']}
    responses = {r['fixture_id']: r for r in llm['records']}
    if (len(statistics) != len(stats['records']) or len(responses) != len(llm['records'])
            or set(statistics) != set(responses)):
        raise ValueError('Current source fixture identities differ or duplicate')
    exposure = llm['exposure']
    if exposure['model'] is not False or exposure['market'] is not False:
        raise ValueError('This archive does not describe an independent LLM line')
    generated = stats['generated_at']
    llm_generated = llm['generated_at']
    as_of = comparison['assembled_at']
    versions = {'next_simple10': [], 'next_rich10': []}
    for row in comparison['records']:
        fid = row['fixture']['match_id']
        original = statistics[fid]
        response = responses[fid]
        if row['fixture'] != original['fixture'] or response['kickoff_utc'] != row['fixture']['kickoff']:
            raise ValueError('Archived fixture identities or kickoff changed between sources')
        fixture = Fixture(**(row['fixture'] | {'kickoff': _time(row['fixture']['kickoff'], 'kickoff')}))
        for cohort, compared_key, source_key in (
                ('next_simple10', 'current_simple', 'refreshed_simple'),
                ('next_rich10', 'current_rich_candidate', 'rich_candidate')):
            probabilities = original[source_key]['probabilities']
            if row[compared_key]['probabilities'] != probabilities or row['current_llm']['probabilities'] != response['probabilities']:
                raise ValueError('Displayed and original archived probabilities disagree')
            stat = {'fixture_id': fid, 'id': f'{cohort}:{fid}:{generated}',
                    'probabilities': probabilities, 'generated_at': generated,
                    'model': 'fixed_ExtraTrees121' if cohort == 'next_rich10' else 'regularized_goal_alpha.01_half_life180'}
            lm = {'fixture_id': fid, 'id': f'llm_20261026:{fid}:{llm_generated}',
                  'probabilities': response['probabilities'], 'generated_at': llm_generated,
                  'model': f"archived subjective LLM; reported model_id={llm['model_id']}",
                  'exposed_to_statistical': False, 'exposed_to_market': False}
            record = multi_record(fixture, as_of, tracks={'statistical': stat, 'llm_independent': lm},
                missing_reasons={'market': 'No new matching market observation was archived.',
                    'llm_model_informed': 'No model-informed response was generated for this version.',
                    'joint_pool': 'No combined probability was generated for this version.'})
            record['source_data_as_of'] = stats['as_of']
            record['source_paths'] = [str(folder / '完整预测与来源.json'), str(folder / '本次独立LLM预测.json')]
            versions[cohort].append(record)
    return versions


def evaluate(records, results):
    ids = {r['fixture']['match_id'] for r in records}
    scores = score_multi(records, {k: v for k, v in results.items() if k in ids})
    # Own coverage is descriptive; comparisons below continue to use shared fixtures.
    scores['own_coverage_metrics'] = {
        arm: _summary([r for r in scores['per_fixture'] if arm in r['scores']], (arm,))['metrics'][arm]
        for arm in TRACKS}
    return scores


def across_versions(left, right, left_scores, right_scores, arm):
    """Fixed named-version comparison, with a distinct match denominator."""
    by_left = {r['fixture_id']: r for r in left_scores['per_fixture'] if arm in r['scores']}
    by_right = {r['fixture_id']: r for r in right_scores['per_fixture'] if arm in r['scores']}
    common = sorted(set(by_left) & set(by_right))
    return {'left': left, 'right': right, 'arm': arm, 'n': len(common), 'fixture_ids': common,
            'left_metrics': _summary([by_left[k] for k in common], (arm,))['metrics'][arm],
            'right_metrics': _summary([by_right[k] for k in common], (arm,))['metrics'][arm],
            'purpose': 'same fixtures, named archived versions; not additional independent observations'}


def run(root, output, additional_cohorts=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = dict(SOURCES)
    for name, source in (additional_cohorts or {}).items():
        if name in {*sources, 'next_simple10', 'next_rich10'}:
            raise ValueError('Additional cohort name must not replace an archived version')
        sources[name] = source
    payloads = {name: read(root / source) for name, source in sources.items()}
    cohorts = {name: doc['records'] for name, doc in payloads.items()}
    cohorts.update(current_cohorts(root))
    results = {}
    for doc in payloads.values():
        for fid, value in doc.get('results', {}).items():
            if fid in results and (results[fid]['home_goals'], results[fid]['away_goals']) != (value['home_goals'], value['away_goals']):
                raise ValueError('Existing result observations conflict')
            results.setdefault(fid, value)
    scored = {name: evaluate(rows, results) for name, rows in cohorts.items()}
    comparisons = []
    for left, right, arms in (
            ('original10', 'personnel_revision9', ('llm_independent', 'joint_pool')),
            ('next_original10', 'next_context_revision7', ('llm_independent', 'joint_pool')),
            ('next_original10', 'next_simple10', ('statistical',)),
            ('next_simple10', 'next_rich10', ('statistical',))):
        comparisons.extend(across_versions(left, right, scored[left], scored[right], arm) for arm in arms)
    if 'next_allcomp_repaired10' in scored:
        comparisons.append(across_versions('next_rich10', 'next_allcomp_repaired10',
            scored['next_rich10'], scored['next_allcomp_repaired10'], 'statistical'))
    if 'next_planned_cup_llm10' in scored:
        comparisons.append(across_versions('next_rich10', 'next_planned_cup_llm10',
            scored['next_rich10'], scored['next_planned_cup_llm10'], 'llm_independent'))
    if 'next_planned_cup_llm10' in scored and 'next_availability_llm10' in scored:
        comparisons.append(across_versions('next_planned_cup_llm10', 'next_availability_llm10',
            scored['next_planned_cup_llm10'], scored['next_availability_llm10'], 'llm_independent'))
    unique = {r['fixture']['match_id'] for rows in cohorts.values() for r in rows}
    pending = unique - set(results)
    generated = datetime.now(timezone.utc).isoformat()
    data = {'assembled_at': generated, 'source_cohorts': sources,
            'additional_current_source': str(root / CURRENT),
            'unique_fixtures': len(unique), 'unique_with_archived_results': len(unique & set(results)),
            'unique_without_archived_results': len(pending),
            'distinct_observation_unit': 'fixture; source versions and repeated LLM values do not add matches',
            'cohort_records': cohorts, 'cohort_scores': scored, 'same_fixture_version_comparisons': comparisons,
            'new_forecasts': 0, 'fits': 0, 'parameter_searches': 0, 'llm_calls': 0, 'new_result_lookups': 0,
            'historical_research_not_merged': True,
            'timestamp_note': 'This assembly time is not a new forecast time; original generation times are retained.',
            'results_note': 'Only previously archived results reused. Missing results do not prove matches are unplayed.',
            'scope': 'Named Premier League cohorts only, not an all-sport or ticket-wide census.'}
    write(output / '分组前瞻记录与评价.json', data)
    lines = ['# 真实前瞻评价接续', '',
        '已接通既有评分程序，没有重训或改写预测。原轮首次预测、原轮修订、下一轮原版、资料修订和10月26日简单／丰富候选各自评价；额外接入的资料修复候选继续单列。', '',
        f'这{len(cohorts)}个版本合计仍是{len(unique)}场不同英超比赛；使用已归档结果的{len(unique & set(results))}场，尚无归档结果的{len(pending)}场。版本数、线路数不增加比赛样本数。', '',
        '本轮不查询新赛果。“尚无归档结果”不自动等于尚未开场；下一轮10场按现有赛程记录为北京时间10月31日20:30至11月3日04:00，未重新核实调期。', '',
        '| 分组 | 比赛数 | 已有赛果 | 统计覆盖 | 独立LLM覆盖 | 市场覆盖 | 联合覆盖 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, score in scored.items():
        cov = score['coverage']
        lines.append(f"| {LABELS.get(name, name)} | {score['n_records']} | {score['n_settled']} | " +
                     ' | '.join(str(cov[a]['available']) for a in ('statistical', 'llm_independent', 'market', 'joint_pool'))+' |')
    lines += ['', '## 首次原轮记录：自身覆盖与同场比较', '',
        '| 线路 | 命中／自身已结算覆盖 | LogLoss↓ | Brier↓ |', '|---|---:|---:|---:|']
    for arm, metric in scored['original10']['own_coverage_metrics'].items():
        n = metric['n']
        lines.append(f"| {ARM_LABELS[arm]} | {metric['top1_accuracy']*n:g}/{n} | {metric['mean_log_loss']:.6f} | {metric['mean_brier']:.6f} |")
    revision_text = []
    for pair in comparisons:
        if pair['left'] == 'original10' and pair['n']:
            a, b, n = pair['left_metrics'], pair['right_metrics'], pair['n']
            revision_text.append(f"{ARM_LABELS[pair['arm']]}由{a['top1_accuracy']*n:g}/{n}变为{b['top1_accuracy']*n:g}/{n}，"
                f"LogLoss由{a['mean_log_loss']:.6f}变为{b['mean_log_loss']:.6f}")
    lines += ['', '以上分母不同，不能直接排名。原轮统计与独立LLM同场10场；全五线共同8场。三项单选命中与票面双选覆盖分开，不用“不败”替代单项概率。', '',
        '人员修订只在原9场共同集合与首次版本比较，不能拿修订9场对首次10场。下一轮资料修订实际只有7场，亦不按10场计算。具体同场ID和逐项概率损失已写入JSON。', '',
        '原共同集合中，'+'；'.join(revision_text)+'。单选命中与概率损失均需保留，当前小样本和均值小差不足以证明人员修订有稳定增益，不挑有利指标。', '',
        '## 下一轮已经能检验什么', '',
        '- 10月24日简单统计→10月26日简单统计：同样模型设置，观察资料刷新后的变化；不将这种变化直接归因于某一个因素。',
        '- 同日简单统计→同日丰富候选：同10场、同批LLM、各自统计概率，观察丰富信息关系是否改善；两组复用的LLM不是20份独立判断。',
        '- 每个版本内统计与LLM三项概率的损失、命中和分歧；有市场的旧版另作同场描述，不能将旧缓存冒充10月26日即时市场。',
        '- 当前丰富候选没有已结算前瞻成绩；历史901/1726不并入该批分母。v0.43／v0.44整合尚无这批完整成员输入，不补造它们的当前概率。', '',
        '[已保存的下一轮10场三项概率](../../outputs/当前丰富信息预测_20261026T030805654404Z/本轮预测与优化结论.md)。本轮只接通评价，未重新生成这些概率。', '',
        '## 时间口径修正', '',
        '10月26日统计资料截止为03:08:05 UTC，实际概率完成为03:09:08 UTC；LLM生成03:14:31 UTC，比较表组装03:16:29 UTC。接入评分时保留实际生成时间，资料截止不冒充概率已经发布的时间。新导出的今天时间也不刷新任何预测。', '',
        '时间和LLM暴露状态来自已有记录及调用方声明，普通本地文件不能独立证明当时已向外发布，亦不能保证LLM预训练无污染。现有证据足以进行可追溯的前瞻观察，不应夸大为认证盲测。', '',
        '## 优化的下一步', '',
        '先让已存下一轮预测产生真实反馈，保留每场三项概率和损失；概率可靠性及类别偏差按后续多批累计观察，不凭一场改参。若根据这些比赛研究调整，它们随后属于开发资料，该调整的效果仍要在更晚比赛确认。', '',
        '资料刷新或明确错队、漏赛、时间错误可以及时修正并留新版本；模型关系、融合权重与深度学习等重大变动仍先讨论。现有回测可用于诊断机制，不能通过反复重算同季来增加独立证据。', '',
        f'本轮停止点：已完成{len(cohorts)}个既有版本的接入、同场配对、原轮重算及未来版本待评记录。该评价入口新训练0、调参0、新LLM调用0、新赛果查询0；不新增hash、冻结contract、baseline或gate。所有新增文件均在项目目录。']
    (output / '前瞻评价接续与优化安排.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return {'output': str(output), 'unique_fixtures': len(unique), 'archived_results': len(unique & set(results)),
            'without_archived_results': len(pending), 'cohorts': len(cohorts), 'fits': 0,
            'original_statistical': scored['original10']['own_coverage_metrics']['statistical'],
            'next_rich_settled': scored['next_rich10']['n_settled']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/前瞻评价接续_20261027')
    parser.add_argument('--additional-cohort', action='append', default=[], metavar='NAME=PATH',
                        help='Add a separately labelled saved version; never replace an original cohort')
    args = parser.parse_args()
    extras = {}
    for value in args.additional_cohort:
        name, separator, path = value.partition('=')
        if not separator or not name or not path or name in extras:
            parser.error('Each additional cohort needs a unique NAME=PATH')
        extras[name] = path
    print(json.dumps(run(args.root, args.output, extras), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()

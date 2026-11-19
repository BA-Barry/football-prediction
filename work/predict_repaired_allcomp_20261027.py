"""Apply the existing 161-feature estimator with repaired schedule observations.

No model fitting, hyperparameter learning, LLM call or result settlement. The
old 121 forecasts and missing-schedule 161 diagnostic remain unchanged.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import math

import joblib
import numpy as np

from football_model.all_competition_features import AllCompetitionFeatures
from football_model.domain import Fixture
from football_model.multiline import multi_record, score_multi
from refresh_current_all_competition_20261026 import load_current_schedule_clubs, ALIASES, FPL_ALIASES
from refresh_allcomp_daily_20261027 import ROOT, CUTOFF, DIRECTORY, members, read, write
from run_current_rich_forecasts_20261026 import prob, CHINESE

ORIGINAL = ROOT/'outputs/当前丰富信息预测_20261026T030805654404Z'
OUTPUT = ROOT/'outputs/全赛事缺口修复与预测_20261027'


def main():
    report = read(DIRECTORY/'coverage_report.json')
    if report['missing_official_league_fixtures'] or report['extra_provider_league_fixtures']:
        raise ValueError('Resolve official/provider league identity differences before using repaired input')
    clubs = load_current_schedule_clubs(DIRECTORY/'schedule_clubs.json')
    schedule = AllCompetitionFeatures(clubs)
    features = read(ORIGINAL/'逐场当前特征.json')
    previous = read(ORIGINAL/'完整预测与来源.json')
    previous_by = {r['fixture']['match_id']: r for r in previous['records']}
    model_path = ORIGINAL/'全赛事缺失输入诊断模型.joblib'
    model = joblib.load(model_path)
    dates = [datetime.fromisoformat(r['fixture']['kickoff']).date() for r in features['records']]
    old_pairs = [r['features']['allcomp_features'] for r in features['records']]
    replayed = model.predict_many(old_pairs, dates)
    max_replay_error = max(abs(a-b) for row, pair in zip(features['records'], replayed)
        for a, b in zip(pair, previous_by[row['fixture']['match_id']]['allcomp_missing_input_diagnostic']['rates']))
    if max_replay_error > 1e-12:
        raise AssertionError('Saved 161 model does not reproduce its old diagnostic')

    source_fixtures = read(DIRECTORY/'raw/official_fpl_fixtures.json')
    source_teams = read(DIRECTORY/'raw/official_fpl_teams.json')
    aliases = {FPL_ALIASES.get(m['displayName'], m['displayName']): ALIASES.get(m['displayName'], m['displayName']) for m in members()}
    teams = {r['id']: aliases[r['name']] for r in source_teams['teams']}
    future = {(teams[r['team_h']], teams[r['team_a']],
               datetime.fromisoformat(r['kickoff_time'].replace('Z', '+00:00'))): r
              for r in source_fixtures if r.get('kickoff_time')}
    as_of = datetime.now(timezone.utc)
    pairs, metadata, confirmations = [], [], []
    for r, d in zip(features['records'], dates):
        fid = r['fixture']['match_id']
        old = previous_by[fid]
        kickoff = datetime.fromisoformat(r['fixture']['kickoff'])
        official = future.get((old['home_label'], old['away_label'], kickoff))
        if not official or official.get('started') is not False or kickoff <= as_of:
            raise ValueError('Archived target is not confirmed future and unstarted in current official source')
        confirmations.append({'fixture_id': fid, 'source_fixture_id': official['id'],
                              'kickoff': kickoff.isoformat(), 'started': official['started']})
        own = r['features']['feature_history']['history']['home']['team']
        away = r['features']['feature_history']['history']['away']['team']
        load = schedule.for_match(home=own, away=away, competition='E0', cutoff=CUTOFF, match_date=d)
        side_pairs = []
        for p, side in zip(r['features']['performance_features'], ('home', 'away')):
            extra = dict(load[side+'_features'])
            if extra.pop('allcomp_log1p_forecast_lead_days') != p['forecast_log1p_lead_days']:
                raise ValueError('Existing performance and new schedule use different cutoffs')
            if len(extra) != 40 or set(extra) & set(p):
                raise ValueError('The existing feature sets do not align')
            side_pairs.append(p | extra)
        if any(not load['metadata']['history'][side]['windows']['28']['usable'] for side in ('home', 'away')):
            raise ValueError('Target clubs still lack a usable past 28-day window')
        pairs.append(side_pairs)
        metadata.append(load['metadata'])
    rates = model.predict_many(pairs, dates)
    generated = datetime.now(timezone.utc).isoformat()
    records, detailed = [], []
    largest_change = 0.
    changed_top = 0
    for row, pair, history, new_features in zip(features['records'], rates, metadata, pairs):
        fixture = row['fixture']
        fid = fixture['match_id']
        old = previous_by[fid]
        forecast = prob(pair)
        old_diagnostic = old['allcomp_missing_input_diagnostic']
        delta = [100*(a-b) for a, b in zip(forecast['probabilities'], old_diagnostic['probabilities'])]
        largest_change = max(largest_change, *map(abs, delta))
        changed_top += int(np.argmax(forecast['probabilities']) != np.argmax(old_diagnostic['probabilities']))
        changed_names = sorted({k for before, after in zip(row['features']['allcomp_features'], new_features)
                                for k in after if before.get(k) != after[k]})
        if any(k in row['features']['performance_features'][0] for k in changed_names):
            raise AssertionError('Repair changed an existing performance input')
        track = {'fixture_id': fid, 'id': f'allcomp_repaired_20261027:{fid}:{generated}',
                 'generated_at': generated, 'model': 'saved_ExtraTrees161_20261026; input_coverage_repaired',
                 'probabilities': forecast['probabilities']}
        f = Fixture(**(fixture | {'kickoff': datetime.fromisoformat(fixture['kickoff'])}))
        record = multi_record(f, generated, tracks={'statistical': track},
            missing_reasons={'llm_independent': 'No new LLM prediction for the repaired-data version.',
                'market': 'No matching fresh market observation.',
                'llm_model_informed': 'No new model-informed LLM prediction.',
                'joint_pool': 'No combined probability generated for the repaired-data version.'})
        record.update(feature_before_date=CUTOFF.isoformat(), source_model_path=str(model_path),
                      source_schedule_path=str(DIRECTORY/'schedule_clubs.json'), model_fitted_this_run=False)
        records.append(record)
        detailed.append({'fixture': fixture, 'home_label': old['home_label'], 'away_label': old['away_label'],
            'unchanged_rich121': old['rich_candidate'], 'old161_missing_schedule': old_diagnostic,
            'new161_repaired_schedule': forecast, 'same_model_change_pp': delta,
            'changed_input_names': changed_names, 'schedule_history': history,
            'features': new_features, 'new_forecast_generated_at': generated, 'actual_outcome': None})
    scores = score_multi(records, {})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT/'标准前瞻记录.json', {'pilot_id': 'allcomp-input-repair-20261027', 'assembled_at': generated,
        'records': records, 'results': {}, 'scope': 'saved 161 model with repaired past schedule input only',
        'model_fits': 0, 'parameter_searches': 0, 'llm_calls': 0})
    write(OUTPUT/'完整对照与输入.json', {'generated_at': generated, 'original_statistical_generated_at': previous['generated_at'],
        'source_data_cutoff': CUTOFF.isoformat(), 'source_model_path': str(model_path),
        'source_coverage_report': str(DIRECTORY/'coverage_report.json'),
        'class_order': ['home', 'draw', 'away'], 'records': detailed, 'confirmed_unstarted_targets': confirmations,
        'model_fits': 0, 'parameter_searches': 0, 'new_forecasts': len(records),
        'old_missing_input_replay_max_rate_error': max_replay_error,
        'same161_top1_changed_count': changed_top, 'max_single_probability_change_pp': largest_change,
        'limitation': 'Same-model missing-vs-observed input sensitivity, not verified accuracy gain, causal fatigue estimate or certified exhaustive competition coverage.'})
    write(OUTPUT/'初始评分.json', scores)
    lines = ['# 全赛事资料缺口修复与独立前瞻', '',
        '已修复宽日期接口400问题：改为13类正式赛事逐日请求，并与20支球队的全赛事接口核对。364个单日响应全部有效；过去28天48场去重正式比赛，30场英超与官方FPL日期和对阵一致，冲突0。20队的28天窗口由未知恢复可用。', '',
        '窗口为UTC日期2026-09-28至2026-10-26之前，保持与原预测相同的资料截止。两场仅在俱乐部接口出现的友谊赛不计入正式负荷。成功请求只确认供应商所覆盖的赛事和日期，不能证明全世界赛事穷尽完整。', '',
        f"新预测于北京时间{datetime.fromisoformat(generated).astimezone(timezone(timedelta(hours=8))):%Y-%m-%d %H:%M:%S}生成。原121预测和原161缺失输入诊断保持不变；今天新取得的资料不回写成原生成时点已知。10场已重新按官方来源确认尚未开赛、对阵与开球时间一致。", '',
        '本轮不训练：直接加载10月26日已有161项模型，先重放其缺失输入版本（10场最大率值差0），再仅替换40项全赛事输入。121项表现、进球／xG基专家、模型参数和比分公式均未改变。', '',
        '| 北京时间 | 比赛 | 原121主／平／客 | 同161缺失输入 | 同161补齐输入 |',
        '|---|---|---:|---:|---:|']
    pct = lambda p: '／'.join(f'{100*v:.1f}' for v in p)
    for r in detailed:
        kickoff = datetime.fromisoformat(r['fixture']['kickoff']).astimezone(timezone(timedelta(hours=8)))
        label = CHINESE[r['home_label']]+'—'+CHINESE[r['away_label']]
        lines.append(f"| {kickoff:%m-%d %H:%M} | {label} | {pct(r['unchanged_rich121']['probabilities'])} | {pct(r['old161_missing_schedule']['probabilities'])} | {pct(r['new161_repaired_schedule']['probabilities'])} |")
    lines += ['', f'同一个161模型补齐资料后，{changed_top}/10场最高单项改变，最大单项概率变化{largest_change:.2f}个百分点。这是输入敏感性，不是命中率提升。原121与161之间还存在训练输入／拟合差异，不能把两者全部差值归因于负荷资料。', '',
        '## 新增资料如何解释', '',
        '每队过去28日正式完赛为3或5场，分别保留过去7／14／28日计数、非联赛场次、此前3场间隔及距离资料截止的间隔。它们是历史赛程描述，不是球员分钟、真实疲劳或已识别战术变化，不手写“多赛一场减几点”。', '',
        '本轮新版本只生成统计概率。未新做LLM、未取得即时赔率、未学习联合权重，其他线路保持缺失；新版本不能继承旧市场缓存并称作即时市场。', '',
        '## 仍然没有覆盖什么', '',
        '英超官方已公布：联赛第8与第9轮之间、10月26日开始的一周安排联赛杯第四轮，13支英超球队参与。该轮处于本次资料截止与目标联赛之间；这是已知计划信息，尚未计入本轮历史完赛特征。[官方赛程说明](https://www.premierleague.com/en/news/4698490/)。', '',
        '当前特征只计算已经完成的比赛，不把计划赛程当已发生负荷，也不估计届时轮换。赛前后续需要真实更新，这份提前数日的预测不能覆盖尚未发生的比赛过程；已公布赛程可以继续作为独立解释资料，不能直接换成手写概率扣分。', '',
        '日期级历史信息和供应商覆盖限制仍在；全赛事输入完整并不保证其关系跨赛季适用。当前无已结算赛果，初始评分0场，不作准确率、价值赔率或串关结论。', '',
        '本轮停止点：已完成单日取数修复、实际窗口核对、原模型重放及10场独立前瞻。新训练0、调参0、新LLM调用0，原预测未修改，全部文件和请求缓存在项目目录。', '',
        '[数据覆盖报告](../../data/current_all_competition/daily_20261027/coverage_report.json)；[完整三项概率与输入](../../outputs/全赛事缺口修复与预测_20261027/完整对照与输入.json)。']
    (OUTPUT/'修复结果与独立预测.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'forecasts': len(records), 'fits': 0,
        'same161_changed_top': changed_top, 'max_probability_change_pp': largest_change,
        'replay_error': max_replay_error, 'settled': scores['n_settled']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

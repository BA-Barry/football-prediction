"""Measure public cross-provider ID reuse, without names or predictive weights."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json

from intake_fpl_availability_20261026 import ROOT, DATA, OUT, write


def main():
    mapping_path = DATA/'id_mapping/Master.csv'
    with mapping_path.open(encoding='utf-8-sig', newline='') as handle:
        mappings = list(csv.DictReader(handle))
    by_code, reverse = defaultdict(list), defaultdict(set)
    for row in mappings:
        if row['code'].isdigit() and row['understat'].isdigit():
            by_code[int(row['code'])].append(row)
            reverse[row['understat']].add(int(row['code']))
    unique = {code: values[0] for code, values in by_code.items()
              if len(values) == 1 and len(reverse[values[0]['understat']]) == 1}
    ambiguous = sorted(set(by_code)-set(unique))
    observations = {}
    for path in (ROOT/'outputs/历史球员使用输入_20261025/历史出场分钟.json',
                 ROOT/'outputs/当前人员历史代表性_20261025/逐场球员使用记录.json'):
        for row in json.loads(path.read_text(encoding='utf-8'))['observations']:
            match_id = row.get('football_data_match_id', row['match_id'])
            key = (match_id, row['team_id'])
            if key in observations and observations[key]['minutes'] != row['minutes']:
                raise ValueError('Overlapping historical minutes disagree')
            observations[key] = row
    histories = defaultdict(list)
    for row in observations.values():
        histories[row['team_id']].append(row)
    for values in histories.values():
        values.sort(key=lambda r: (r['date'], r['match_id']))
    manifest = {r['path']: r for r in json.loads((DATA/'selected_snapshot_manifest.json').read_text(encoding='utf-8'))['observations']}
    with gzip.open(OUT/'逐场赛前状态特征.json.gz', 'rt', encoding='utf-8') as handle:
        rows = [r for r in json.load(handle)['rows'] if r['period'] in ('2024-25', '2025-26')]
    counts, records, samples = defaultdict(Counter), [], []
    for row in rows:
        snapshot = manifest[row['archive_observation']['path']]
        period = row['period']
        season_column = period[2:4]+'-'+period[-2:]
        for team in (row['home'], row['away']):
            earlier = [r for r in histories[team] if r['date'] < row['forecast_before_date']]
            for player in snapshot['teams'][team]['flagged_players']:
                if player['status'] not in ('i', 'd', 's'):
                    continue
                counts[period]['injury_doubtful_suspension_player_contexts'] += 1
                code = player.get('code')
                candidate = unique.get(code)
                agreement = bool(candidate and candidate.get(season_column, '').isdigit()
                                 and int(candidate[season_column]) == player['id'])
                understat_key = None if not agreement else 'understat:player:'+candidate['understat']
                previous = [] if understat_key is None else [r for r in earlier if understat_key in r['minutes']]
                counts[period]['code_has_unique_understat_mapping'] += candidate is not None
                counts[period]['mapping_and_historical_season_id_agree'] += agreement
                counts[period]['joined_with_earlier_same_team_appearances'] += bool(previous)
                counts[period]['joined_without_earlier_same_team_appearances'] += agreement and not previous
                counts[period]['season_id_conflicts_or_unlisted'] += candidate is not None and not agreement
                record = {'match_id': row['match_id'], 'forecast_before_date': row['forecast_before_date'],
                    'team': team, 'fpl_code': code, 'fpl_id_at_snapshot': player['id'],
                    'status_at_snapshot': player['status'], 'understat_player_key': understat_key,
                    'mapping_season_column': season_column, 'same_season_id_agrees': agreement,
                    'earlier_same_team_appearance_n': len(previous),
                    'latest_earlier_appearance_date': previous[-1]['date'] if previous else None}
                records.append(record)
                if len(samples) < 5 and previous:
                    samples.append(record | {'display_name': player['web_name']})
    summary = {'created_at': datetime.now(timezone.utc).isoformat(),
        'source_url': 'https://github.com/ChrisMusson/FPL-ID-Map/blob/main/Master.csv',
        'source_csv': str(mapping_path), 'mapping_rows': len(mappings),
        'unique_one_to_one_code_understat_rows': len(unique), 'ambiguous_codes_excluded': ambiguous,
        'historical_team_observations': len(observations), 'target_matches': len(rows),
        'joining': 'FPL stable code plus same-season numeric FPL ID agreement; no name/fuzzy join; one-to-one rows only',
        'earlier_history': 'already-saved same-team Understat appearances with date strictly before each target forecast cutoff',
        'by_period': {k: dict(v) for k, v in counts.items()}, 'examples': samples,
        'scope': 'identity/coverage preparation only; not a player ability estimate, loss coefficient or prediction',
        'licensing': 'The mapping repository has no explicit LICENSE according to source review; factual ID table retained for personal research, not claimed as open-licensed project code',
        'limitations': ['current retrospective ID reconciliation is not a historic player-availability announcement',
            'community ID mapping is not an official identity guarantee',
            'same-team EPL appearances omit cups, foreign leagues and new signings without earlier EPL records',
            'minutes are source minute-level usage records, not seconds, skill or replacement ability',
            'unmapped/ambiguous/conflicting players remain unknown; no zero contribution assumption'],
        'new_model_fits': 0, 'current_forecasts_replaced': False}
    target = ROOT/'outputs/FPL球员身份连接_20261026'
    write(target/'连接覆盖实测.json', summary)
    write(target/'逐场有标记球员身份连接.json', {'records': records})
    lines = ['# 球员身份桥梁：复用现成编号表', '',
        '复用[ChrisMusson/FPL-ID-Map](https://github.com/ChrisMusson/FPL-ID-Map/blob/main/Master.csv)的事实编号表。'
        '通过稳定FPL code连接Understat ID，并核对各历史赛季FPL数字ID；不按姓名猜测，不把每季FPL数字ID跨季复用。', '',
        '| 赛季 | 伤病／存疑／停赛球员情境 | 编号可一致连接 | 另有更早同队出场记录 |',
        '|---|---:|---:|---:|']
    for period, c in summary['by_period'].items():
        lines.append(f"| {period} | {c['injury_doubtful_suspension_player_contexts']} | {c['mapping_and_historical_season_id_agree']} | {c['joined_with_earlier_same_team_appearances']} |")
    lines += ['', '一个球员在不同比赛前重复出现会计为多个情境；这些数不是独立球员数或独立训练样本数。'
        '有更早出场记录仅表示历史参与可观测，不代表能力已知、当前仍效力本队或预计缺席。', '',
        '未连接或没有更早同队记录的球员保留未知，不设影响为零。当前只完成身份与覆盖测量，'
        '没有训练球员效果、设置缺阵扣分或修改比赛概率。', '',
        '该表仓库未提供明确LICENSE；本轮仅为个人研究保留事实编号，未将项目称为获得明确开源许可。', '',
        '本节点在真实数据连接与覆盖报告完成后停止；球员贡献与替补能力属于后续独立建模选择。']
    (target/'连接结果与下一步.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

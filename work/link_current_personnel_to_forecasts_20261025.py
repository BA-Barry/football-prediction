"""Attach newly measured past personnel use to saved future forecasts."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/当前人员历史代表性_20261025'
OLD = ROOT/'outputs/历史代表性与证据联合_20261025/current/逐场证据与分歧.json'
ALIASES = {
    'Manchester United': 'Man United', 'AFC Bournemouth': 'Bournemouth',
    'Leeds United': 'Leeds', 'Nottingham Forest': "Nott'm Forest",
    'Coventry City': 'Coventry', 'Hull City': 'Hull', 'Ipswich Town': 'Ipswich',
    'Manchester City': 'Man City', 'Brighton & Hove Albion': 'Brighton',
    'Tottenham Hotspur': 'Tottenham', 'Crystal Palace': 'Crystal Palace',
    'Aston Villa': 'Aston Villa', 'Newcastle United': 'Newcastle',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compact(profile):
    starter = profile['starting_appearance_overlap_last5_vs_preceding15']
    minutes = profile['source_minute_overlap_last5_vs_previous_season_last15']
    return {'team': profile['team'], 'before_date': profile['before_date'],
        'available_matches_by_season': profile['available_matches_by_season'],
        'missing_match_ids': profile['missing_match_ids'],
        'independent_verified_matches_by_season': profile.get('independent_verified_matches_by_season'),
        'starter_share_overlap_recent5_vs_preceding15': starter['overlap'],
        'starter_window_missing_reason': starter['missing_reason'],
        'source_minute_overlap_recent5_vs_previous_season_last15': minutes['overlap'],
        'minute_window_missing_reasons': minutes['missing_reasons'],
        'current_minute_share_from_players_observed_previous_season_same_team':
            profile['current_season_minute_share_by_previously_observed_same_team_players'],
        'recent_window': {k: minutes['recent'][k] for k in ('n_matches', 'date_min', 'date_max')},
        'previous_season_reference': {k: minutes['previous_season_last15'][k] for k in ('n_matches', 'date_min', 'date_max')},
        'recent_players': [{k: r[k] for k in ('player_id', 'source_names',
            'recent_source_minute_share', 'previous_season_same_team_status',
            'previous_season_same_team_observed_positive_matches', 'previous_season_observed_team_ids')}
            for r in profile['recent_players']],
        'numerical_probability_effect': None,
        'interpretation': 'Past domestic league participation, not ability, injury status, target lineup or transfer registry'}


def pct(value):
    return '未知' if value is None else f'{value:.1%}'


def main():
    now = datetime.now(timezone.utc)
    source = read(OUT/'球队历史代表性.json')
    profiles = {r['team']: r for r in source['teams']}
    cases = []
    for old in read(OLD)['cases']:
        fixture = old['fixture']
        if datetime.fromisoformat(fixture['kickoff']) <= now:
            continue
        sides = {}
        for side in ('home', 'away'):
            name = fixture[side]
            key = 'football-data:E0:'+ALIASES.get(name, name)
            # Explicit name aliases only: no fuzzy join or zero-filled missing club.
            sides[side] = compact(profiles[key]) if key in profiles else {
                'team': key, 'missing_reason': 'no_explicit_team_profile', 'numerical_probability_effect': None}
        cases.append({'fixture': fixture, 'original_record_as_of': old['original_record_as_of'],
            'original_tracks': old['tracks'], 'current_personnel_context': sides,
            'original_forecasts_changed': False, 'new_combined_probabilities': None,
            'use': 'New model-informed interpretation; not a new independent LLM forecast or an original input'})
    result = {'prepared_at': now.isoformat(), 'new_personnel_measurement_at': source['summary']['measured_at'],
        'past_match_cutoff_exclusive': source['summary']['before_date'],
        'source': str(OUT/'球队历史代表性.json'), 'original_forecast_source': str(OLD),
        'coverage_summary': source['summary'], 'cases': cases,
        'policy': ['Keep original probabilities and forecasts unchanged.',
            'A high overlap cannot establish high ability, accuracy or independent confirmation.',
            'A player not previously observed at the same club has an unknown impact, not a zero or negative impact.',
            'A missing promoted-club reference is missing coverage, not 100% personnel turnover.',
            'This newly acquired evidence cannot be backdated as an original forecast input.']}
    (OUT/'十场人员依据更新.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# 十场未来预测的人员依据更新', '',
        '原统计与独立LLM概率仍为10月24日已保存版本；本轮新增截至10月24日的历史人员使用信息。',
        '下表的比例不是胜率；不会据此自动升降原预测，也不会产生融合概率。', '',
        '| 比赛 | 主队本季分钟来自上季同队出现过球员 | 客队同指标 | 主/客近5场分钟对上季末15场重叠 |',
        '|---|---:|---:|---|']
    for case in cases:
        f, sides = case['fixture'], case['current_personnel_context']
        key = 'current_minute_share_from_players_observed_previous_season_same_team'
        overlap = 'source_minute_overlap_recent5_vs_previous_season_last15'
        h, a = sides['home'], sides['away']
        lines.append(f"| {f['home']}—{f['away']} | {pct(h.get(key))} | {pct(a.get(key))} | {pct(h.get(overlap))} / {pct(a.get(overlap))} |")
    lines += ['', '每个比例的完整球员ID、原始分钟、参考比赛和缺失原因见JSON及球队报告。', '',
        'LLM可以据此收回缺乏人员依据的“延续上季实力”或“阵容大幅削弱”等表述；仍不能从人员重叠推断损失多少进球或概率。', '',
        '对于上季没有英超同队记录的Coventry、Hull、Ipswich，当前缺的是跨级历史，而不是已经证明全队更换。', '',
        '球队模型仍根据已发生比赛学习当前攻防状态。人员使用信息有助于识别历史解释的边界；是否需要改变状态更新方式，须与球员能力、对手和实际表现一起确定。', '',
        '本轮没有重新结算预测赛果，没有拟合参数或更改既有预测。']
    (OUT/'十场人员依据更新.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'cases': len(cases), 'prepared_at': now.isoformat(),
        'output': str(OUT/'十场人员依据更新.json'), 'probabilities_changed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()

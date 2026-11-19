"""Summarize the already downloaded historical schedule evidence, without network access."""
from datetime import datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

BASE = REPOSITORY_ROOT / Path('outputs/全赛事赛程可用性_20261025')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    rows, all_ids, club_ids, receipts, conflicts, special = [], set(), set(), [], [], []
    for year in range(2022, 2026):
        season = f'{year}-{str(year+1)[-2:]}'
        folder = BASE/f'{season}全量'
        summary = read(folder/'相邻赛季补全_全量覆盖与负荷测量.json')
        events = read(folder/'相邻赛季补全_去重历史赛事.json')['events']
        membership = read(folder/'历史参赛身份目录.json')['clubs']
        all_ids.update(e['event_id'] for e in events)
        club_ids.update(c['team_id'] for c in membership)
        for receipt in (folder/'receipts').glob('*.json'):
            value = read(receipt)
            if value.get('retrieved_at'):
                receipts.append(value['retrieved_at'])
        conflicts.extend({'season': season, **c} for c in summary['incomplete_source_status_conflicts'])
        special.extend({'season': season, **c} for c in summary['known_specials'])
        row = {k: summary[k] for k in (
            'clubs', 'unique_events', 'unique_competitive_completed_events',
            'unique_friendly_exhibition_events', 'league_exact_fixture_checks',
            'league_expected_FD_matches', 'league_completed_score_verified',
            'league_ordinary_goal_target_verified', 'ordinary_score_disagreements',
            'load_comparison', 'earliest_event', 'latest_event')}
        row.update({'season': season, 'incomplete_status_count': len(summary['incomplete_source_status_conflicts']),
            'known_special_count': len(summary['known_specials']),
            'membership_exact': all(not m['missing_independent_labels'] and not m['unexpected_source_labels']
                and not m['duplicate_normalized_labels'] for m in summary['membership'].values()),
            'missing_fixture_count': len(summary['missing_FD_matches']),
            'unclassified_or_join_issue_count': len(summary['issues']),
            'duplicate_payload_conflict_count': len(summary['source_duplicate_conflicts']),
            'summary_file': str(folder/'相邻赛季补全_全量覆盖与负荷测量.json'),
            'events_file': str(folder/'相邻赛季补全_去重历史赛事.json'),
            'load_file': str(folder/'相邻赛季补全_联赛逐队负荷观察.json')})
        rows.append(row)
    sum_keys = ('clubs', 'unique_events', 'unique_competitive_completed_events',
        'unique_friendly_exhibition_events', 'league_exact_fixture_checks', 'league_expected_FD_matches',
        'league_completed_score_verified', 'league_ordinary_goal_target_verified', 'ordinary_score_disagreements')
    totals = {k: sum(r[k] for r in rows) for k in sum_keys}
    totals['unique_club_ids'] = len(club_ids)
    totals['global_unique_event_ids'] = len(all_ids)
    totals['load_comparison'] = {k: sum(r['load_comparison'][k] for r in rows)
        for k in rows[0]['load_comparison']}
    output = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_retrieval_time_utc_min': min(receipts),
        'source_retrieval_time_utc_max': max(receipts), 'seasons': rows, 'totals': totals,
        'incomplete_source_status_conflicts': conflicts, 'existing_goal_fit_exclusions_preserved': special,
        'scope': '2022/23 through 2025/26 completed historical five-league club memberships; dates from 2022-07-01 through 2026-06-30 only',
        'all_competition_completeness_certified': False, 'current_model_inputs_modified': False,
        'new_models_fitted': False, 'current_2026_27_results_requested': False,
        'important_use_limits': [
            'Fixture identity coverage is not completed-match validation: six ESPN status conflicts are retained explicitly',
            'Three already excluded resumed/administratively awarded fixtures remain excluded from ordinary goal targets',
            'Abandoned and resumed match sessions are not reconstructed; source dates are not guaranteed actual finish dates',
            'Cup/friendly completeness, original announcement time and original kickoff-time publication are not certified',
            'No next-match pressure, exact rest hours, extra-time duration or 90-minute cup scores inferred',
            'Prior-only source-completed-event gaps are descriptive coverage measurements, not evidence of predictive improvement']}
    (BASE/'四季覆盖汇总.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    table = '\n'.join(f"|{r['season']}|{r['clubs']}|{r['unique_events']}|{r['unique_competitive_completed_events']}|{r['league_exact_fixture_checks']}/{r['league_expected_FD_matches']}|{r['league_completed_score_verified']}|{r['incomplete_status_count']}|" for r in rows)
    load_table = '\n'.join(f"|{r['season']}|{r['load_comparison']['with_previous_league_in_window']}|{r['load_comparison']['previous_formal_match_is_nonleague']}|{r['load_comparison']['all_kickoff_gap_le_72h']}|{r['load_comparison']['league_only_kickoff_gap_le_72h']}|" for r in rows)
    report = f'''# 四季历史全赛事赛程接入实测

本轮已下载并检查2022/23—2025/26五大联赛386个俱乐部赛季的历史赛程。按赛事ID去重后得到{totals['global_unique_event_ids']}场历史赛事，其中{totals['unique_competitive_completed_events']}场被来源标记为已完成的正式比赛。联赛对阵身份7082/7082与现有Football-Data逐场对应；这不等于7082场都可用作完整90分钟目标。严格按来源已完成状态可确认7076场，另有6场状态问题；再保留原先3场复赛／判罚赛果排除后，普通进球目标交叉确认7073场。

原始响应、请求地址、取回时间、逐队观察和测量结果均在本目录。取回时间范围：{min(receipts)}—{max(receipts)}（UTC）。只读取2022年7月至2026年6月的历史窗口，未请求2026/27赛果；未读取首发，未改动当前模型输入，也未训练新模型。

## 范围与覆盖

|赛季|俱乐部|去重赛事|来源已完成正式赛事|联赛身份对应|来源已完成比分对应|状态待解|
|---|---:|---:|---:|---:|---:|---:|
{table}

四季参赛名单均来自带赛季的ESPN核心目录，再与Football-Data当季球队核对，名单缺失／多余均为0。普通联赛比分不一致为0；重复赛事载荷冲突为0。各季完整结果见`相邻赛季补全_全量覆盖与负荷测量.json`；原始单赛季结果仍保留。友谊赛／表演赛单列，附加赛单列，不将它们混为常规联赛场次。

## 确实补到了什么信息

下表只使用更早、被来源明确标记已完成的正式赛事；按开球到开球间隔测量，不将间隔称为终场后的真实休息时长。同一球队只看联赛会漏掉杯赛和欧战。

|赛季|可比球队比赛观察|上一场正式赛为非联赛|全赛事间隔≤72小时|只看联赛间隔≤72小时|
|---|---:|---:|---:|---:|
{load_table}

合计{totals['load_comparison']['with_previous_league_in_window']}条可比观察中，{totals['load_comparison']['previous_formal_match_is_nonleague']}条的上一场是杯赛／欧战等非联赛。这是实际缺失信息的测量，不是疲劳会降低多少胜率的结论，也不证明模型已经提升。赛季窗口开头缺更早来源，不能把无前序记录解释为充分休息。

## 六场来源状态问题，必须保留

2022/23西甲5场通过有明确日期的每日端点补到了比分与对阵，均与Football-Data相符，但来源同时返回`STATUS_SCHEDULED`、`completed=false`、时钟90分钟：2023-02-11阿尔梅里亚—贝蒂斯、塞维利亚—马略卡、瓦伦西亚—毕尔巴鄂；2023-02-13西班牙人—皇家社会；2023-02-15皇马—埃尔切。这属于来源状态矛盾，不能靠比分相同推断其元数据已可信；本次严格的历史负荷观察未将它们作为完成场次。

2025/26南特—图卢兹（2026-05-17，事件746714）每日响应为0-0，但`STATUS_ABANDONED`、`completed=false`、时钟22分钟。Football-Data现存0-0且没有既有进球训练排除标志。此场不能按普通完整90分钟0-0评估；当前只是报告冲突，未擅自改动原数据或当前模型。原始证据：`2025-26全量/raw/daily_F1_20260517.json`。不得把同为0-0写成验证通过。

此外，原有格拉纳达—毕尔巴鄂、乌迪内斯—罗马复赛及柏林联合—波鸿行政判罚3项进球训练排除继续保留。它们的分段开赛、复赛时间及真正踢球时长尚未重建，负荷应用也应单独处理。

## 接入时修正的具体陷阱

- 普通球队列表的`season`参数没有确保历史参赛身份，已改用带年份的核心目录；历史响应顶层的当前积分和战绩字段一律不作为特征。
- 单一`season=2024`响应漏掉2025年6月世俱杯；用已下载相邻历史赛季、按实际日期取并集，补到29场该窗口内世俱杯正式赛。未请求2026赛季实时数据。
- 阿森纳2023年MLS全明星赛事按表演赛；塞维利亚2023年UEFA-CONMEBOL Club Challenge按官方说明归为友谊性质；意甲2023年斯佩齐亚—维罗纳保级附加赛不计入380场常规联赛。
- 根据明确`homeAway`认定主客，不按数组顺序猜。赛程响应的中立场布尔值与终场时钟并不充分可靠；点球阶段也不能证明曾踢加时。

## 下一步能做、还不能声称的事

数据可作为下一轮广泛赛前方案的历史赛程输入候选，与球队强度、近期攻防、射门／xG等一同检验，仍先做前推验证。不要预先写死“休息少三天减多少胜率”。需要在真正使用时明确处理6个状态矛盾、3项特殊赛果及赛季起点缺失。

尚未独立证明所有杯赛／友谊赛完全无缺；没有重建每个历史时点当时已公布的后续赛程，因此不生成未来赛程压力；没有精确终场时间、加时时长或杯赛90分钟比分，因此不造时长负荷或杯赛进球目标。最后一个历史赛季没有请求2026赛季来源，其自然年友谊赛覆盖可能不全。

## 主要来源

采集方式参考[SoccerData ESPN实现](https://github.com/probberechts/soccerdata/blob/master/soccerdata/espn.py)，实际数据保留ESPN原始响应与每次请求地址。例如：[2024英超历史参赛目录](https://sports.core.api.espn.com/v2/sports/soccer/leagues/eng.1/seasons/2024/teams?limit=1000)、[南特—图卢兹所在历史每日赛程](https://site.api.espn.com/apis/site/v2/sports/soccer/fra.1/scoreboard?dates=20260517)。端点为公开可读接口，没有据此宣称官方稳定API或无限采集授权。

赛事分类及窗口补全依据：[UEFA Club Challenge性质说明](https://www.uefa.com/uefaconmebolclubchallenge/news/0283-1878f90ea07e-365545745d98-1000--lowdown-uefa-conmebol-club-challenge/)、[切尔西官方2025世俱杯赛程](https://www.chelseafc.com/en/news/article/chelseas-fifa-club-world-cup-fixtures-confirmed)。[openfootball英格兰](https://github.com/openfootball/england)与[欧战库](https://github.com/openfootball/champions-league)可作后续独立日期核对来源，未把未实测仓库覆盖率写成已验证结果。
'''
    (BASE/'四季全赛事接通与使用限制.md').write_text(report, encoding='utf-8')
    print(json.dumps({'totals': totals, 'status_conflicts': len(conflicts), 'existing_exclusions': len(special),
        'report': str(BASE/'四季全赛事接通与使用限制.md'), 'summary': str(BASE/'四季覆盖汇总.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()

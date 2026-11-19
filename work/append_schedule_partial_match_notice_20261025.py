"""Append official partial-match evidence without recalculating earlier coverage artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path

BASE = Path('G:/Codex/football-prediction-2026-10-22/outputs/全赛事赛程可用性_20261025')
report_path = BASE/'四季全赛事接通与使用限制.md'
notice_path = BASE/'官方部分比赛补充说明_20261025.json'
observed_at = datetime.now(timezone.utc).isoformat()
evidence = {
    'verified_at_utc': observed_at,
    'purpose': 'Official historical partial-match evidence added after the preceding provider-coverage measurement',
    'previous_coverage_outputs_recalculated': False,
    'current_model_inputs_modified_by_this_script': False,
    'important_interpretation': 'Provider completed=true and matching final scores do not establish 90 minutes actually played. Earlier counts measure provider/FD consistency under the then-current parser, not a physical-duration certificate.',
    'matches': [
        {'match_id': 'football-data:F1:2025/26:Nantes|Toulouse', 'event_id': '746714',
         'date': '2026-05-17', 'accepted_score': [0, 0],
         'provider_status': 'STATUS_ABANDONED', 'provider_completed': False, 'provider_clock_seconds': 1320,
         'official_evidence': [
             {'url': 'https://ligue1.com/fr/articles/l1_article_5119-', 'publication_date': '2026-05-17',
              'finding': 'Ligue 1 reports definitive abandonment shortly after minute20, following crowd incidents and a security decision.'},
             {'url': 'https://ligue1.com/fr/articles/l1_article_5189-discipline-toutes-les-decisions-l1-270526',
              'publication_date': '2026-05-28', 'decision_date': '2026-05-27',
              'finding': 'Disciplinary commission accepted the0-0 score at abandonment; acceptance does not turn the played interval into90minutes.'}],
         'use_limit': 'Do not use accepted0-0 as an ordinary completed90-minute goal target; event remains evidence of a partial playing session.'},
        {'match_id': 'football-data:F1:2024/25:Montpellier|St Etienne', 'event_id': '706656',
         'date': '2025-03-16', 'accepted_score': [0, 2],
         'provider_status': 'STATUS_FULL_TIME', 'provider_completed': True, 'provider_clock_seconds': 3600,
         'provider_evidence_files': [str(BASE/'2024-25全量/raw/team_178.json'), str(BASE/'2024-25全量/raw/team_274.json')],
         'official_evidence': [
             {'url': 'https://www.asse.fr/actualites/le-score-est-enterine-les-trois-points-acquis--ac38439',
              'publication_date': '2025-04-02',
              'finding': 'Club confirms match was not restarted and the existing0-2 was accepted by the disciplinary commission.'},
             {'url': 'https://ligue1.com/fr/articles/l1_article_2269-le-point-sur-la-lutte-pour-le-maintien-020425-l1',
              'publication_date': '2025-04-02',
              'finding': 'League identifies interruption at minute57 on March16 and subsequent disciplinary loss for Montpellier.'}],
         'use_limit': 'Provider completed=true / FULL_TIME means settlement here, not90minutes played. Do not use as ordinary90-minute goal target or invent90-minute load.'}
    ]}
notice_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
marker = '## 官方补充：完成标志不代表踢满90分钟'
notice = '''

## 官方补充：完成标志不代表踢满90分钟

2026-10-25补充核实：上文7076场“来源已完成比分对应”、7073场“普通进球目标对应”是当次接口／Football-Data及当时解析规则的一致性测量，不能解释为已经独立证明踢满90分钟。本补充保留原始覆盖数字和文件，不静默改写既有测量；新增的部分比赛例外由中央数据解析单独处理。

- 南特—图卢兹（2026-05-17，事件746714）：[法甲官方比赛公告](https://ligue1.com/fr/articles/l1_article_5119-)确认比赛在20分钟后不久被永久中止；[纪律决定](https://ligue1.com/fr/articles/l1_article_5189-discipline-toutes-les-decisions-l1-270526)确认接受当时0-0。0-0是被认可的结算比分，不是完整90分钟进球观察。
- 蒙彼利埃—圣埃蒂安（2025-03-16，事件706656）：[法甲官方说明](https://ligue1.com/fr/articles/l1_article_2269-le-point-sur-la-lutte-pour-le-maintien-020425-l1)明确比赛在第57分钟中断，随后按纪律决定确认蒙彼利埃失利；[圣埃蒂安官方公告](https://www.asse.fr/actualites/le-score-est-enterine-les-trois-points-acquis--ac38439)确认不再恢复比赛并认可当时0-2。现存赛程响应却是`completed=true`、`STATUS_FULL_TIME`、时钟60分钟，说明完成标志与比分相同仍不足以证明90分钟实际暴露时间。

两场均须从普通90分钟进球目标中单独识别，不能将行政确认的结算结果当作全场实际进球过程。两场确实发生过的部分比赛也不能简单写成90分钟负荷；本轮未重建实际净比赛时长或精确结束时间。上文7073个旧规则普通目标含新增发现的蒙彼利埃例外，不能继续称为7073场已验证的完整90分钟比赛；最终可用训练数以中央解析修正后的明确规则及重新测量为准。详细证据见`官方部分比赛补充说明_20261025.json`。
'''
text = report_path.read_text(encoding='utf-8')
top_warning = '> 2026-10-25官方补充：来源完成标志和比分一致不等于真实踢满90分钟；另发现蒙彼利埃—圣埃蒂安的部分比赛例外。原覆盖数字保留其当次测量含义，使用前请阅读文末官方补充。\n\n'
if marker not in text:
    first, rest = text.split('\n', 1)
    text = first+'\n\n'+top_warning+rest.lstrip('\n')+notice
    report_path.write_text(text, encoding='utf-8')
print(json.dumps({'report': str(report_path), 'official_notice': str(notice_path), 'coverage_files_rewritten': False}, ensure_ascii=False))

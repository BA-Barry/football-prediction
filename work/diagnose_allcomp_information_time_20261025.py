"""Read-only retrospective schedule information-age diagnostic; no refitting."""
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
from project_paths import PROJECT_ROOT as REPOSITORY_ROOT

import numpy as np

from football_model.all_competition_features import AllCompetitionFeatures, LEAGUE_SLUGS

ROOT = REPOSITORY_ROOT
OUT = ROOT/'outputs/完整清理与全赛事联合模型_20261025T064817840440Z'
WEEKDAYS = ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')


def distribution(values):
    if not values:
        return {'n': 0, 'median': None, 'p90': None, 'max': None}
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9)), 'max': float(np.max(values))}


def summarize(rows):
    return {'team_contexts': len(rows),
        'cutoff_weekday_counts': dict(Counter(r['cutoff_weekday'] for r in rows)),
        'forecast_lead_days': distribution([r['lead_days'] for r in rows]),
        'latest_usable_before_cutoff_gap_to_target_days': distribution([r['latest_known_to_target_days'] for r in rows if r['latest_known_to_target_days'] is not None]),
        'latest_observed_before_target_gap_days_diagnostic_only': distribution([r['latest_retrospective_to_target_days'] for r in rows if r['latest_retrospective_to_target_days'] is not None]),
        'latest_known_nonleague_n': sum(r['latest_known_nonleague'] is True for r in rows),
        'latest_known_usable_n': sum(r['latest_known_to_target_days'] is not None for r in rows),
        'latest_known_nonleague_fraction_among_usable': sum(r['latest_known_nonleague'] is True for r in rows)/sum(r['latest_known_to_target_days'] is not None for r in rows) if any(r['latest_known_to_target_days'] is not None for r in rows) else None,
        'excluded_between_cutoff_and_target_formal_events_total': sum(r['excluded_between_n'] for r in rows),
        'team_contexts_with_excluded_between_events': sum(r['excluded_between_n'] > 0 for r in rows),
        'excluded_between_count_per_team_context': distribution([r['excluded_between_n'] for r in rows]),
        'source_member_scope_contains_full_cutoff_to_target': sum(r['scope_covers_cutoff_and_target'] for r in rows),
        'unresolved_between_cutoff_and_target_total': sum(r['unresolved_between_n'] for r in rows)}


def main():
    builder = AllCompetitionFeatures.from_directory(ROOT/'outputs/全赛事赛程可用性_20261025')
    source = json.loads((OUT/'完整历史特征.json').read_text(encoding='utf-8'))['rows']
    observations = []
    for row in source:
        cutoff, target = map(date.fromisoformat, (row['fit_before_date'], row['date']))
        for side in ('home','away'):
            club = builder._clubs[(row['competition'], row[side])]
            metadata = row['allcomp_history']['history'][side]
            scope = next(((s,e) for s,e in club.coverage if s <= cutoff <= e), None)
            known = metadata['last_completed_event']
            known_usable = bool(known and known['usable'])
            before_target = [e for e in club.events if e.category == 'competitive' and e.match_date < target
                             and scope is not None and scope[0] <= e.match_date < scope[1]]
            good = [e for e in before_target if e.completed and e.exclusion_reason is None]
            between = [e for e in good if cutoff <= e.match_date < target]
            unresolved = [e for e in before_target if cutoff <= e.match_date < target
                          and (not e.completed or e.exclusion_reason is not None)]
            retrospective = good[-1] if good else None
            observations.append({'match_id': row['match_id'], 'period': row['period'], 'side':side,
                'team': row[side], 'cutoff': cutoff.isoformat(), 'target_date':target.isoformat(),
                'cutoff_weekday': WEEKDAYS[cutoff.weekday()], 'lead_days': (target-cutoff).days,
                'latest_known_to_target_days': (target-date.fromisoformat(known['date'])).days if known_usable else None,
                'latest_known_nonleague': (known['competition_slug'] != LEAGUE_SLUGS[row['competition']]
                    or metadata['last_completed_event']['event_id'] in {
                        e.event_id for e in club.events if not e.is_regular_league_fixture}) if known_usable else None,
                'latest_retrospective_to_target_days': (target-retrospective.match_date).days if retrospective else None,
                'excluded_between_n':len(between),
                'excluded_between_event_ids': [e.event_id for e in between],
                'excluded_between_event_dates': [e.match_date.isoformat() for e in between],
                'unresolved_between_n':len(unresolved),
                'scope_covers_cutoff_and_target': bool(scope and scope[0] <= cutoff <= target < scope[1])})
    by_period = {p:summarize([r for r in observations if r['period'] == p]) for p in sorted({r['period'] for r in observations})}
    by_weekday = {w:summarize([r for r in observations if r['cutoff_weekday'] == w]) for w in WEEKDAYS if any(r['cutoff_weekday'] == w for r in observations)}
    result = {'measured_at_utc': datetime.now(timezone.utc).isoformat(), 'purpose':'retrospective diagnostic of fixed28day cutoff information age; not predictive inputs',
        'fixture_contexts':len(source), 'overall':summarize(observations), 'by_period':by_period, 'by_cutoff_weekday':by_weekday,
        'observations': [
            'All four periods have complete requested member scope over cutoff through target; source-scope gaps do not explain this large between-period contrast.',
            'August1 plus28day cutoffs preserve the same weekday within each season, but differ across seasons: Monday, Tuesday, Thursday, Friday.',
            'European cup matches commonly sit between domestic weekends, so the latest strictly-before-cutoff event can depend strongly on the weekday; this is an information-time composition issue.',
            'Season and cutoff weekday are confounded here. Different club/competition schedules also change by year, so this study cannot estimate a causal weekday effect or prove all of the nonleague contrast is due to the anchor.',
            'Retrospective events in [cutoff,target) were correctly excluded from model inputs, but their counts show that original28day updates can omit substantial actually completed schedule history near kickoff.'],
        'next_design_suggestions':[
            'Keep slower team-parameter refits separate from fast date/count feature refreshes; using yesterday-available schedule data need not require daily retraining of every team model.',
            'For an earlier-prediction product specify one lead, such as1or3 calendar days, rebuild every historical training example at that same lead, then retain past-only fitting.',
            'Compare fixed28day features and a predeclared fixed-lead refresh on common historical fixtures; learn choices on the earlier period, not by selecting later-season winners.',
            'Continue withholding future fixture pressure until announcement/rescheduling availability can be represented; historical eventual schedules cannot stand in for what was known then.'],
        'limits': ['date-level source history, not exact T24h timestamps', 'completed source events are counts, not measured minutes or causal fatigue',
                   'member-scope coverage still does not certify every competition was captured',
                   'diagnostic looks at historical events between cutoff and target; they are never added to the running model'],
        'modules_or_model_inputs_changed':False, 'new_model_fits':0, 'current_2026_27_results_reviewed':False}
    (OUT/'信息时点与负荷覆盖诊断.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'信息时点诊断逐队日期引用.json').write_text(json.dumps({'rows':observations}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k in ('fixture_contexts','overall','by_period')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

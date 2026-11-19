"""Measure cutoff-aligned historical schedule inputs; no goals/models fitted."""
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path

from football_model.all_competition_features import AllCompetitionFeatures

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'outputs/全赛事赛程可用性_20261025'
OUT = ROOT/'outputs/全赛事负荷特征接入_20261025'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    builder = AllCompetitionFeatures.from_directory(SOURCE)
    old = json.loads((ROOT/'outputs/多信息多算法前推_20261025T060704977450Z/历史特征样本.json').read_text(encoding='utf-8'))['rows']
    contexts, errors, counts = [], [], Counter()
    by_period, max_gap = {}, 0.
    for row in old:
        cutoff = date.fromisoformat(row['fit_before_date'])
        ctx = builder.for_match(home=row['home'], away=row['away'], competition=row['competition'],
            cutoff=cutoff, match_date=date.fromisoformat(row['date']))
        group = by_period.setdefault(row['period'], Counter())
        group['fixture_contexts'] += 1
        for side in ('home', 'away'):
            features = ctx[f'{side}_features']
            history = ctx['metadata']['history'][side]
            counts['team_contexts'] += 1
            for prefix, key in [('identity_missing', 'identity_available'),
                                ('scope_missing', 'scope_contains_cutoff')]:
                if features[f'own_allcomp_{key}'] == 0:
                    counts[prefix] += 1
                    group[prefix] += 1
            for days in (7, 14, 28):
                if features[f'own_allcomp_window_usable_{days}d'] == 0:
                    counts[f'unusable_{days}d'] += 1
                    group[f'unusable_{days}d'] += 1
                w = history['windows'][str(days)]
                for text in w['selected_event_dates']:
                    if text >= cutoff.isoformat():
                        errors.append({'id': row['match_id'], 'reason': 'future window event'})
                counts[f'recorded_references_{days}d'] += len(w['selected_event_ids'])
                if days == 28 and w['excluded_or_unresolved_events']:
                    counts['partial_or_status_affected_28d'] += 1
            last = history['last_completed_event']
            if last:
                if last['date'] >= cutoff.isoformat():
                    errors.append({'id': row['match_id'], 'reason': 'future latest event'})
                if last['usable']:
                    counts['usable_latest_event'] += 1
                    if last['competition_slug'] not in ('eng.1', 'ger.1', 'esp.1', 'ita.1', 'fra.1'):
                        counts['latest_event_nonleague'] += 1
                        group['latest_event_nonleague'] += 1
                    max_gap = max(max_gap, features['own_allcomp_last_log1p_gap_to_cutoff_days'])
            contexts.append({'match_id': row['match_id'], 'side': side,
                'cutoff': cutoff.isoformat(), 'features': features, 'history': history})
    keys = sorted(contexts[0]['features'])
    result = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': builder.source_metadata, 'historical_target_rows': len(old),
        'scope': 'schedule features only for archived6990 contexts; the new clean-model runner removes incomplete goal targets separately',
        'feature_keys_per_scoring_side': keys, 'feature_count_per_scoring_side': len(keys),
        'counts': dict(counts), 'by_period': {k: dict(v) for k, v in by_period.items()},
        'max_log1p_latest_gap': max_gap, 'errors': errors, 'new_model_fits': 0,
        'cutoff_rule': 'all histories strictly earlier than original28daycutoff; windows relative to cutoff, lead separately',
        'limits': ['date-level retrospective calendar, no first-publication timestamps',
            'member-source scope is not certified complete all-competition coverage',
            'no future fixture pressure, fulltime scores, lineups, actual minutes or fatigue multipliers',
            'full-count windows with partial or unresolved competitive events remain missing',
            'source members absent from a season do not acquire fabricated zero-load histories']}
    write(OUT/'全赛事负荷特征覆盖.json', result)
    write(OUT/'逐场负荷特征与引用.json', {'rows': contexts})
    print(json.dumps({k: v for k,v in result.items() if k not in ('source', 'feature_keys_per_scoring_side')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

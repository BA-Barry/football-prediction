"""Independent arithmetic replay of clean integrated schedule inputs.

Reads original member event observations; does not import the schedule builder
or refit any predictive model. Never queries current match results.
"""
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

ROOT = Path('G:/Codex/football-prediction-2026-10-22')
SOURCE = ROOT/'outputs/全赛事赛程可用性_20261025'
OUT = ROOT/'outputs/完整清理与全赛事联合模型_20261025T064817840440Z'
SLUGS = {'E0': 'eng.1', 'D1': 'ger.1', 'SP1': 'esp.1', 'I1': 'ita.1', 'F1': 'fra.1'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    seasons = read(SOURCE/'四季覆盖汇总.json')['seasons']
    blocked, clubs, cache = {}, {}, {}
    for info in seasons:
        for item in read(info['summary_file']).get('known_specials', []):
            blocked[str(item['event_id'])] = item['reason']
    for item in read(SOURCE/'官方部分比赛补充说明_20261025.json')['matches']:
        blocked[str(item['event_id'])] = 'officially_confirmed_partial_match'
    for info in seasons:
        year = int(info['season'].split('-')[0])
        path = Path(info['events_file']).with_name('相邻赛季补全_逐队历史观察.json')
        for source_id, club in read(path)['clubs'].items():
            identity = (club['competition'], f"football-data:{club['competition']}:{club['independent_label']}")
            target = clubs.setdefault(identity, {'source_id': source_id, 'intervals': [], 'events': {}})
            assert target['source_id'] == source_id
            target['intervals'].append((date(year, 7, 1).toordinal(), date(year+1, 7, 1).toordinal()))
            for row in club['events']:
                # Only dates/competition/status/identity enter the independent replay.
                # Source score, minutes, lineups and target result are never read.
                event_id = str(row['event_id'])
                stamp = date.fromisoformat(row['date_utc_text'][:10])
                reason = blocked.get(event_id)
                if row['category'] == 'competitive' and row['completed'] is not True:
                    reason = reason or 'unresolved_source_completion_status'
                value = {'id': event_id, 'day': stamp.toordinal(), 'date': stamp.isoformat(),
                    'slug': row['competition_slug'], 'category': row['category'],
                    'completed': row['completed'], 'regular': row['is_regular_league_fixture'], 'reason': reason}
                assert event_id not in target['events'] or target['events'][event_id] == value
                target['events'][event_id] = value
    for club in clubs.values():
        spans = []
        for start, end in sorted(club['intervals']):
            if spans and spans[-1][1] >= start:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
        club['intervals'] = spans
        club['events'] = sorted(club['events'].values(), key=lambda e: (e['day'], e['id']))

    def replay(comp, team, cutoff):
        key = (comp, team, cutoff)
        if key in cache:
            return cache[key]
        club = clubs.get((comp, team))
        span = next(((s,e) for s,e in club['intervals'] if s <= cutoff <= e), None) if club else None
        prior = [r for r in club['events'] if span and span[0] <= r['day'] < cutoff and r['category'] == 'competitive'] if club else []
        good = [r for r in prior if r['completed'] is True and r['reason'] is None]
        bad = [r for r in prior if r['completed'] is not True or r['reason'] is not None]
        f = {'identity_available': float(club is not None), 'scope_contains_cutoff': float(span is not None),
             'covered_days_within_28': float(min(28, cutoff-span[0])) if span else 0.}
        m = {'team': team, 'source_team_id': club['source_id'] if club else None,
             'contiguous_member_scope': [date.fromordinal(v).isoformat() for v in span] if span else None,
             'cutoff': date.fromordinal(cutoff).isoformat(), 'windows': {}}
        for w in (7,14,28):
            covered = bool(span and span[0] <= cutoff-w)
            selected = [r for r in good if r['day'] >= cutoff-w]
            unresolved = [r for r in bad if r['day'] >= cutoff-w]
            usable = covered and len(unresolved) == 0
            f[f'window_usable_{w}d'] = float(usable)
            f[f'formal_count_{w}d'] = float(len(selected)) if usable else None
            f[f'nonleague_count_{w}d'] = float(sum(r['slug'] != SLUGS[comp] or not r['regular'] for r in selected)) if usable else None
            f[f'unresolved_count_{w}d'] = float(len(unresolved)) if covered else None
            m['windows'][str(w)] = {'covered': covered, 'usable': usable,
                'selected_event_ids': [r['id'] for r in selected], 'selected_event_dates': [r['date'] for r in selected],
                'excluded_or_unresolved_events': [{'event_id': r['id'], 'date': r['date'], 'reason': r['reason']} for r in unresolved]}
        last = good[-1] if good else None
        last_ok = last is not None and all(r['day'] < last['day'] for r in bad)
        f['last_log1p_gap_to_cutoff_days'] = math.log1p(cutoff-last['day']) if last_ok else None
        f['last_is_nonleague'] = float(last['slug'] != SLUGS[comp] or not last['regular']) if last_ok else None
        recent = good[-3:]
        recent_ok = len(recent) == 3 and all(r['day'] < recent[0]['day'] for r in bad)
        f['previous3_count'] = float(len(recent)) if span else None
        f['previous3_usable'] = float(recent_ok)
        f['previous3_log1p_mean_interval_days'] = math.log1p((recent[-1]['day']-recent[0]['day'])/2) if recent_ok else None
        m['last_completed_event'] = {'event_id': last['id'], 'date': last['date'], 'competition_slug': last['slug'], 'usable': last_ok} if last else None
        m['previous3_events'] = [{'event_id': r['id'], 'date': r['date']} for r in recent]
        m['previous3_usable'] = recent_ok
        cache[key] = (f,m)
        return f,m

    rows = read(OUT/'完整历史特征.json')['rows']
    errors, counts, by_period = [], Counter(), {}
    max_delta = 0.
    removed = ('cutoff_staleness_days', 'previous_league_interval_days', 'known_league_matches_before_cutoff_7d',
               'known_league_matches_before_cutoff_14d', 'known_league_matches_before_cutoff_28d', 'same_league_gap_to_match_days')
    for row in rows:
        cutoff = date.fromisoformat(row['fit_before_date']).toordinal()
        expected = [replay(row['competition'], row[s], cutoff) for s in ('home','away')]
        group = by_period.setdefault(row['period'], Counter())
        group['fixtures'] += 1
        for side, label in enumerate(('home','away')):
            counts['scoring_sides'] += 1
            own, meta = expected[side]
            other, _ = expected[1-side]
            features = {**{f'own_allcomp_{k}':v for k,v in own.items()}, **{f'opponent_allcomp_{k}':v for k,v in other.items()}}
            actual = row['allcomp_features'][side]
            if row['allcomp_history']['history'][label] != meta:
                errors.append({'id': row['match_id'], 'side': label, 'issue': 'reference metadata mismatch'})
            for key, value in features.items():
                saved = actual.get(key)
                delta = 0. if value is None and saved is None else abs(saved-value) if value is not None and saved is not None else math.inf
                max_delta = max(max_delta, delta)
                counts['feature_values_compared'] += 1
                if delta > 1e-12:
                    errors.append({'id': row['match_id'], 'side':label, 'key':key, 'expected':value, 'saved':saved})
            lead = math.log1p(date.fromisoformat(row['date']).toordinal()-cutoff)
            counts['feature_values_compared'] += 1
            if actual['forecast_log1p_lead_days'] != lead or row['performance_features'][side]['forecast_log1p_lead_days'] != lead:
                errors.append({'id': row['match_id'], 'issue': 'shared forecast lead mismatch'})
            for field in removed:
                if any(f'{prefix}_{field}' in actual or f'{prefix}_{field}' in row['performance_features'][side] for prefix in ('own','opponent')):
                    errors.append({'id': row['match_id'], 'issue': 'old partial schedule input retained', 'field':field})
            if own['identity_available'] == 0:
                counts['identity_missing'] += 1
            if own['scope_contains_cutoff'] == 0:
                counts['scope_missing'] += 1
            for w in (7,14,28):
                window = meta['windows'][str(w)]
                counts[f'event_references_{w}d'] += len(window['selected_event_ids'])
                if own[f'window_usable_{w}d'] == 0:
                    counts[f'unusable_{w}d'] += 1
                    group[f'unusable_{w}d'] += 1
                for event_id, stamp in zip(window['selected_event_ids'], window['selected_event_dates']):
                    if date.fromisoformat(stamp).toordinal() >= cutoff or event_id in blocked:
                        errors.append({'id': row['match_id'], 'issue': 'future or partial event used', 'event_id':event_id})
            if meta['last_completed_event'] and meta['last_completed_event']['usable']:
                counts['usable_latest_event'] += 1
                if own['last_is_nonleague'] == 1:
                    counts['latest_event_nonleague'] += 1
                    group['latest_event_nonleague'] += 1
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': not errors,
        'source_files': [str(Path(s['events_file']).with_name('相邻赛季补全_逐队历史观察.json')) for s in seasons],
        'method': 'independent ordinal-day arithmetic from original member observations; no AllCompetitionFeatures import and no model fitting',
        'clean_contexts': len(rows), 'counts': dict(counts), 'by_period': {k:dict(v) for k,v in by_period.items()},
        'max_numeric_delta': max_delta, 'errors': errors, 'blocked_events': blocked,
        'checks': ['all40 schedule values per scoring side and shared lead independently replayed',
            'all saved source reference metadata matched the original event data',
            'counts and latest/last3 use only competitive events on dates strictly earlier than cutoff',
            'original12 own/opponent partial-league schedule keys absent from rich models',
            'partial/resumed/status-conflict windows retain explicit missing counts',
            'source goals, lineups and actual minutes never read by schedule replay'],
        'limitations': ['date-level source history only; publication or rescheduling timestamps not established',
            'provider member-season scope does not certify all competitions were complete',
            'this checks inputs, not predictive accuracy or causal fatigue effects'], 'new_model_fits': 0}
    (OUT/'全赛事特征独立核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','checks','limitations')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

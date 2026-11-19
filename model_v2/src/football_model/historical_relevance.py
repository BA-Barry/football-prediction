"""Observed historical personnel usage, without target-lineup or strength claims.

The two windows never overlap. This is descriptive starting-appearance share,
not minutes, a transfer register, causal ability, or a calibrated uncertainty.
Source first-publication times remain the caller's separate responsibility.
"""
from collections import Counter
from datetime import date
import math
import re


def _date(value):
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise ValueError('Use an ISO calendar date')
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('Use an ISO calendar date')
    return parsed


def _window(rows):
    counts = Counter(player for row in rows for player in row['starters'])
    total = sum(counts.values())
    return {
        'n_matches': len(rows), 'match_ids': [r['match_id'] for r in rows],
        'date_min': rows[0]['date'] if rows else None,
        'date_max': rows[-1]['date'] if rows else None,
        'seasons': sorted({r['season'] for r in rows}),
        'players': sorted(counts), 'starting_appearances': dict(sorted(counts.items())),
        'total_starting_appearances': total,
        'shares': {p: counts[p]/total for p in sorted(counts)} if total else {},
    }


def personnel_usage_overlap(observations, team, before_date, *, recent_matches=5,
                            reference_matches=15):
    """Compare last five past XIs with the *preceding*, disjoint fifteen.

    Only rows dated strictly before ``before_date`` are inspected for starters.
    A match on/after that cutoff cannot change the returned measurement. Both
    windows must reach their requested sample sizes for overlap to be numeric;
    partial windows retain actual observations but return None, never zero.
    Player IDs must be provider:player:id, all from one provider namespace.
    """
    cutoff = _date(before_date)
    if not isinstance(team, str) or ':' not in team:
        raise ValueError('Use an explicit namespaced team identity')
    for count in (recent_matches, reference_matches):
        if type(count) is not int or count <= 0:
            raise ValueError('Window sizes must be positive integers')
    history, seen, providers = [], set(), set()
    for raw in observations:
        played = _date(raw['match_date'])
        if played >= cutoff:
            continue
        side = 'home' if raw['home'] == team else 'away' if raw['away'] == team else None
        if side is None:
            continue
        ident = raw['match_id']
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('Earlier team fixtures need unique explicit match IDs')
        seen.add(ident)
        players = raw[side+'_starters']
        if (not isinstance(players, (list, tuple)) or len(players) != 11
                or any(not isinstance(p, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+:player:[^:\s]+', p)
                       for p in players) or len(set(players)) != 11):
            raise ValueError('Observed XI needs 11 unique namespaced player IDs')
        providers.update(p.split(':', 1)[0] for p in players)
        history.append({'match_id': ident, 'date': played.isoformat(),
                        'season': raw['season'], 'starters': list(players)})
    if len(providers) > 1:
        raise ValueError('Do not compare provider ID spaces without an explicit mapping')
    history.sort(key=lambda r: (r['date'], r['match_id']))
    recent_rows = history[-recent_matches:]
    stop = max(0, len(history)-recent_matches)
    reference_rows = history[max(0, stop-reference_matches):stop]
    recent, reference = _window(recent_rows), _window(reference_rows)
    complete = len(recent_rows) == recent_matches and len(reference_rows) == reference_matches
    overlap = None
    reference_seen = recent_seen = None
    if complete:
        rp, qp = recent['shares'], reference['shares']
        overlap = math.fsum(min(rp.get(p, 0), qp.get(p, 0)) for p in set(rp)|set(qp))
        reference_seen = math.fsum(v for p,v in qp.items() if p in rp)
        recent_seen = math.fsum(v for p,v in rp.items() if p in qp)
    return {
        'team': team, 'before_date': cutoff.isoformat(),
        'requested_windows': {'recent': recent_matches, 'preceding_reference': reference_matches},
        'past_matches_available': len(history), 'recent': recent, 'preceding_reference': reference,
        'overlap': overlap,
        'reference_start_share_seen_in_recent': reference_seen,
        'recent_start_share_seen_in_reference': recent_seen,
        'missing_reason': None if complete else 'insufficient_disjoint_past_windows',
        'latest_observation_age_days': ((cutoff-_date(history[-1]['date'])).days if history else None),
        'interwindow_gap_days': ((_date(recent_rows[0]['date'])-_date(reference_rows[-1]['date'])).days
                                 if recent_rows and reference_rows else None),
        'windows_span_seasons': len(set(recent['seasons'])|set(reference['seasons'])) > 1,
        'provider_namespace': next(iter(providers)) if providers else None,
        'interpretation': 'observed starting-appearance share overlap; not minutes retention, transfers, current roster or strength',
        'availability': 'strictly earlier calendar dates; original publication/revision times not established',
    }

"""Explain existing 1X2 disagreements without pretending to fuse evidence.

The caller supplies source-linked LLM reviews. They are interpretations, not
trained effects or authenticated facts. Shared ranks never increase a numeric
probability, and marginal forecast spans are not confidence intervals.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

from .batch_multiline import _fixture
from .multiline import _line, build_packet
from .prospective import OUTCOMES, _text, _time


ISSUE_KINDS = {'shared_information', 'unmodelled_context', 'historical_transfer',
               'unquantified_effect', 'source_conflict', 'unsupported_inference'}


def _tops(probabilities):
    largest = max(probabilities)
    return [OUTCOMES[i] for i, value in enumerate(probabilities) if value == largest]


def reconcile_evidence(record, packet, review, *, as_of):
    """Return a source-linked interpretation of two original forecasts.

    No target results or retrospective probability revisions are accepted.
    ``packet`` is the original independent LLM packet, including descriptive
    context and news. ``review`` is explicitly model-informed criticism.
    """
    fixture = _fixture(record['fixture'])
    at = _time(as_of, 'reconciliation as_of')
    if at >= _time(fixture.kickoff, 'kickoff'):
        raise ValueError('Reconciliation as_of must precede kickoff')
    if _time(record['as_of'], 'original record as_of') > at:
        raise ValueError('Cannot examine a record created after this review')
    if packet['fixture'] != record['fixture']:
        raise ValueError('Review packet and original forecast fixture disagree')
    if _time(packet['as_of'], 'original packet as_of') > at:
        raise ValueError('Review packet was not yet available')
    if not isinstance(review, Mapping) or review.get('fixture_id') != fixture.match_id:
        raise ValueError('A review must explicitly identify this fixture')
    reviewed_at = _time(review.get('generated_at'), 'LLM review generated_at')
    if not _time(packet['as_of'], 'packet as_of') <= reviewed_at <= at:
        raise ValueError('Review must follow the packet and precede its analysis')
    if _time(record['as_of'], 'record as_of') > reviewed_at:
        raise ValueError('Review cannot precede the saved record it examines')
    # This run reuses the original packet, not a subsequently enriched corpus.
    source_packet = build_packet(fixture, packet['as_of'], packet.get('evidence', []),
        packet.get('unknowns', []), descriptive_context=packet.get('descriptive_context', []))
    available = {r['id']: r for r in source_packet['evidence']+source_packet['descriptive_context']}
    if review.get('exposed_to_statistical') is not True:
        raise ValueError('Reconciliation criticism must disclose model exposure')
    if review.get('exposed_to_market') is not False:
        raise ValueError('This review does not use the market track')
    if 'probabilities' in review or 'probability_adjustment' in review:
        raise ValueError('A qualitative review is not a new numerical forecast')
    issues = []
    raw_issues = review.get('issues')
    if not isinstance(raw_issues, list):
        raise ValueError('Supply the source-linked issue list explicitly')
    seen = set()
    for issue in raw_issues:
        ident = _text(issue.get('id'), 'issue id')
        if ident in seen:
            raise ValueError('Issue ids must be unique within a fixture')
        seen.add(ident)
        kind = issue.get('kind')
        if kind not in ISSUE_KINDS:
            raise ValueError('Unknown evidence interpretation kind')
        refs = issue.get('evidence_ids')
        if not isinstance(refs, list) or not refs or any(r not in available for r in refs):
            raise ValueError('Every interpretation must reference available packet evidence')
        issues.append({'id': ident, 'kind': kind,
            'summary': _text(issue.get('summary'), 'issue explanation'),
            'evidence_ids': list(dict.fromkeys(refs)),
            'numerical_effect': None,
            'next_question': _text(issue.get('next_question'), 'next question'),
            'assessment_origin': 'LLM model-informed interpretation; unvalidated'})
    stat = _line(record['tracks'].get('statistical'), 'statistical', fixture, reviewed_at)
    llm = _line(record['tracks'].get('llm_independent'), 'llm_independent', fixture, reviewed_at)
    comparison = None
    if stat is not None and llm is not None:
        p, q = stat['probabilities'], llm['probabilities']
        tops, topq = _tops(p), _tops(q)
        comparison = {
            'statistical_top_outcomes': tops, 'llm_top_outcomes': topq,
            'common_top_outcomes': [o for o in OUTCOMES if o in tops and o in topq],
            'same_unique_top': len(tops) == len(topq) == 1 and tops == topq,
            'llm_minus_statistical': dict(zip(OUTCOMES, [b-a for a,b in zip(p,q,strict=True)])),
            'total_variation': math.fsum(abs(b-a) for a,b in zip(p,q,strict=True))/2,
            'between_forecast_span': {name: [min(a,b), max(a,b)]
                for name,a,b in zip(OUTCOMES,p,q,strict=True)},
            'span_interpretation': 'Descriptive bounds of two forecasts; not a confidence interval, posterior or a new probability vector',
            'agreement_interpretation': 'Shared top rank is not independent corroboration, a >50% claim or evidence of calibration',
        }
    return {
        'fixture': deepcopy(record['fixture']), 'analyzed_at': at.isoformat(),
        'original_record_as_of': record['as_of'],
        'original_packet_as_of': packet['as_of'],
        'tracks': {'statistical': stat, 'llm_independent': llm},
        'comparison': comparison, 'issues': issues,
        'review': {'generated_at': reviewed_at.isoformat(),
            'model': _text(review.get('model'), 'review model description'),
            'exposed_to_statistical': True, 'exposed_to_market': False,
            'scope': 'explanation of saved forecasts, not an independent new forecast'},
        'combined_probabilities': None,
        'quantified_update_status': 'No fitted personnel/context effect supplied; original numerical forecasts retained',
        'current_personnel_continuity': None,
        'current_personnel_continuity_reason': 'Not supplied by this descriptive packet; do not infer roster continuity from status/news lists',
        'original_predictions_changed': False,
        'limits': ['News is the original cached observation, not refreshed current availability.',
            'Source-linked interpretations can still be wrong; references do not prove a causal effect.',
            'Player status u is not an injury; cumulative season minutes do not measure expected minutes or player ability.',
            'Unresolved history transfer must not be disguised by taking a mean or by increasing certainty.'],
    }

"""Describe what archived personnel news mentions; never estimate strength.

Labels classify text, not verified medical states. Source status, source chance,
expected return wording and news metadata must keep their separate meanings.
"""
import re


RULES = (
    ('medical_reference', 'medical_phrase',
     r'\b(injur(?:y|ies)|illness|infection|concussion|hernia|strain|sprain|dislocation|'
     r'anterior cruciate ligament|dead leg|gashed leg|heart condition)\b|^(?:Knock|Pelvic|Neck)$'),
    ('fitness_reference', 'fitness_phrase', r'\b(lack of match fitness|fatigue|reconditioning)\b'),
    ('suspension_reference', 'suspension_phrase', r'\b(suspended|suspension|banned|ban)\b'),
    ('eligibility_reference', 'eligibility_phrase',
     r'\b(ineligible|international clearance|work permit|not registered)\b'),
    ('transfer_reference', 'transfer_phrase',
     r'^(?:(?:has\s+)?(?:joined|joins|rejoined)|permanent transfer|permanent move|'
     r'loaned|loan to|on loan|(?:on\s+a\s+)?season[-\s]long\s+loan|'
     r'half[-\s]season\s+loan|one[-\s]year\s+loan(?:\s+deal)?|'
     r'signed (?:for|by)|transferred|transfer to|released|retired|recalled by|'
     r'returned to .*parent club|left the club|left club|left by mutual consent|'
     r'left .* on |(?:has\s+)?departed the club|moved to|'
     r'contract (?:terminated|cancelled|ended)|mutual termination of contract|'
     r'loan (?:ended|terminated)|end of loan)\b'),
    ('nonmedical_absence_reference', 'nonmedical_phrase',
     r'\b(personal reasons|international duty|family reasons|bereavement|paternity)\b'),
    ('selection_restriction_reference', 'selection_phrase',
     r'\b(not being considered for selection|not training with his club until further notice|'
     r'made unavailable for selection|(?:won[\u0027\u2019]t|will not) be considered for selection)\b'),
)


def extract_personnel_news(raw_news, raw_status):
    """Return source-preserving text annotations; unknown text stays unknown.

    ``raw_status`` does not determine any text label. A reference to an injury
    does not assert that it is active now, causes this absence or loses goals.
    Return/suspension-until text is not turned into an observed end date or year.
    Transfer labels include loan/contract/departure references, without verifying
    the event. Bare anatomical source labels are not a clinical diagnosis.
    """
    if raw_news is not None and not isinstance(raw_news,str):
        raise ValueError('News must be source text or null')
    if raw_status is not None and not isinstance(raw_status,str):
        raise ValueError('Status must be source text or null')
    text = '' if raw_news is None else raw_news
    reason = text.split(' - ',1)[0].strip()
    labels, matches = [], []
    for category, rule, pattern in RULES:
        found = re.search(pattern,reason,flags=re.I)
        if found:
            labels.append(category)
            matches.append({'rule_id':rule,'category':category,'source_span':found.group(0)})
    empty = not text.strip()
    unspecified = reason.lower() in ('unknown','unspecified','not known')
    if empty:
        labels=['no_news_text']
    elif unspecified:
        labels.append('unspecified_reason_text')
    elif not labels:
        labels=['unclassified_text']
    timing = [m.group(0) for m in re.finditer(
        r'\b(?:Expected back[^\n]*|Return date (?:unknown|not known)\b|'
        r'(?:unknown|not known) return date\b|Out for[^\n]*|(?:Suspended|Suspension) until[^\n]*)',
        text,flags=re.I)]
    percentages = [{'source_span':m.group(0),'number_as_written':m.group(1)}
                   for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%',text)]
    return {'raw_news':raw_news,'raw_status':raw_status,
        'text_categories':labels,'matched_source_spans':matches,
        'reason_segment':reason if reason else None,
        'expected_return_wording':timing,'source_percentage_mentions':percentages,
        'change_or_negation_language':bool(re.search(
            r'\b(no longer|no|not|never|won[\u0027\u2019]t|can[\u0027\u2019]t|cannot|returned|'
            r'returning|recovered|available|unavailable|lifted|fit to)\b',text,flags=re.I)),
        'classification_method':'transparent text rules, LLM-reviewed for wording boundaries; not fitted from match outcomes',
        'clinical_state_confirmed':False,'actual_target_absence_confirmed':False,
        'normalized_return_date':None,'estimated_strength_loss':None,
        'limits':['Text reference is not confirmed current injury, severity, eligibility or availability.',
            'Raw status does not supply a diagnosis; u/n are not injury labels.',
            'Source percentage is not a calibrated participation, starting or winning probability.',
            'Expected return/suspension-until wording is not an observed recovery/end date; no missing year is inferred.',
            'Change/negation language is a cue only; it does not resolve whether any condition or restriction is active.',
            'Bare anatomical labels and medical references do not supply a clinical diagnosis.',
            'Transfer references include loans, recalls, contracts and departures; the event is not verified.']}

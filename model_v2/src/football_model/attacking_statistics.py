"""Read historical Football-Data attacking counts, without fitting a model.

``non_target_shots`` means HS-HST / AS-AST. The source does not split blocked
shots from shots wide, so this quantity must not be called pure shots wide.
Missing or contradictory counts remain unknown. Original cells are retained.
"""
import re


STATISTIC_COLUMNS = {
    'shots': ('HS', 'AS'),
    'shots_on_target': ('HST', 'AST'),
    'corners': ('HC', 'AC'),
}


def parse_attacking_counts(row):
    counts, issues, raw = {}, [], {}
    for name, columns in STATISTIC_COLUMNS.items():
        values = [row.get(key) for key in columns]
        raw.update(zip(columns, values))
        text = [value.strip() if isinstance(value, str) else '' for value in values]
        if not all(re.fullmatch(r'[0-9]+', value) for value in text):
            counts[name] = None
            issues.append({'statistic': name, 'reason': 'missing_or_invalid_integer',
                           'columns': list(columns)})
        else:
            counts[name] = [int(value) for value in text]
    shots, on_target = counts['shots'], counts['shots_on_target']
    counts['non_target_shots'] = None
    if shots is not None and on_target is not None:
        if any(on > total for on, total in zip(on_target, shots)):
            issues.append({'statistic': 'shot_counts', 'reason': 'on_target_exceeds_total',
                           'columns': ['HS', 'AS', 'HST', 'AST']})
            counts['shots'] = counts['shots_on_target'] = None
        else:
            counts['non_target_shots'] = [total-on for total, on in zip(shots, on_target)]
    return {'counts': counts, 'raw_statistic_cells': raw, 'issues': issues,
            'non_target_semantics': 'total shots minus shots on target; blocked/wide split unavailable'}

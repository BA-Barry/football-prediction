"""Reuse validated old five-match total-xG residual windows as a joint package.

No fitting, source repair, new history selection or network access. Existing
121 side features are copied unchanged; each side receives own/opponent views
of the same eight residual descriptions, for 137 columns in total.
"""
from copy import deepcopy
import math

from football_model.xg_residual_patterns import FEATURE_NAMES, summarize_residual_patterns
from opponent_adjusted_xg_features_20261026 import build_adjusted_rows


ADDED_FEATURE_NAMES = tuple(f'{role}_{name}'
                            for role in ('own', 'opponent') for name in FEATURE_NAMES)


def build_residual_pattern_rows(original_rows, actual_xg_records):
    """Return all original rows with ``residual_pattern_features/history``.

    ``build_adjusted_rows`` first applies its existing exact identity, score,
    last-five and ``fit<=F<played<target_F`` checks. Its rows and means are
    retained as provenance only. Both ``record.xg`` and saved ``xg_rates`` are
    total xG, never source NPxG or a scaled full-goal forecast.
    """
    original_rows = list(original_rows)
    adjusted, old_coverage = build_adjusted_rows(original_rows, actual_xg_records)
    index = {row['match_id']: row for row in original_rows}
    result = []
    maximum_net_difference = 0.
    for original, validated in zip(original_rows, adjusted, strict=True):
        if original['match_id'] != validated['match_id']:
            raise ValueError('Validated residual rows changed order')
        histories, summaries = {}, {}
        for side in ('home', 'away'):
            history = deepcopy(validated['adjusted_xg_residual_history']['history'][side])
            pairs = []
            for observation in history['observations']:
                mid = observation['match_id']
                actual, past = actual_xg_records[mid], index[mid]
                own = 0 if observation['team_was_home'] else 1
                actual_for, actual_against = map(float, (actual.xg[own], actual.xg[1-own]))
                expected_for, expected_against = map(float, (past['xg_rates'][own], past['xg_rates'][1-own]))
                for_residual = actual_for-expected_for
                against_residual = actual_against-expected_against
                net, total = for_residual-against_residual, for_residual+against_residual
                difference = abs(net-observation['team_xg_net_residual'])
                maximum_net_difference = max(maximum_net_difference, difference)
                if difference > 1e-12:
                    raise ValueError('For/against residuals disagree with validated net residual')
                observation.update(actual_for=actual_for, actual_against=actual_against,
                                   expected_for=expected_for, expected_against=expected_against,
                                   residual_for=for_residual, residual_against=against_residual,
                                   net_residual=net, total_residual=total)
                pairs.append(observation)
            complete = not history['unavailable_reasons']
            if complete and len(pairs) != 5:
                raise ValueError('Complete validated window must contain five references')
            summary = summarize_residual_patterns(pairs if complete else None)
            old_mean, new_mean = history['residual_mean'], summary['xg_residual_net_mean_5']
            if ((old_mean is None) != (new_mean is None)
                    or old_mean is not None and not math.isclose(old_mean, new_mean, rel_tol=0., abs_tol=1e-12)):
                raise ValueError('New net mean does not reproduce the existing validated net mean')
            history['summary'] = summary
            histories[side], summaries[side] = history, summary
        pairs = []
        for i, (own, opponent) in enumerate((('home', 'away'), ('away', 'home'))):
            added = {f'own_{key}': value for key, value in summaries[own].items()}
            added.update({f'opponent_{key}': value for key, value in summaries[opponent].items()})
            if set(added) & set(original['fresh_performance_features'][i]):
                raise ValueError('Residual pattern feature names collide with existing inputs')
            pairs.append({**original['fresh_performance_features'][i], **added})
        if any(len(pair) != 137 for pair in pairs):
            raise ValueError('Expected unchanged 121 inputs plus 16 descriptions')
        row = dict(original)
        row['residual_pattern_features'] = pairs
        row['residual_pattern_history'] = {
            'forecast_before_date': original['forecast_before_date'], 'history': histories,
            'unit': 'total_xg_actual_minus_saved_total_xg_expectation',
            'window': 5, 'variance_denominator': 'population_n',
            'missing_policy': 'whole_window_unknown_if_any_original_reference_unavailable',
        }
        result.append(row)
    coverage = deepcopy(old_coverage)
    coverage.update(
        original_side_feature_count=121, adjusted_side_feature_count=137,
        added_feature_names=list(ADDED_FEATURE_NAMES),
        definition='Eight joint total-xG residual descriptions over the unchanged five earlier fixtures, for own/opponent',
        source_unit='actual record.xg total xG and saved past.xg_rates total-xG expectation; not NPxG',
        mean_units='xG', variance_covariance_units='xG_squared',
        maximum_component_net_difference=maximum_net_difference,
        redundant_representation_note='Sixteen side descriptions share information: mean(net)=mean(for)-mean(against), mean(total)=mean(for)+mean(against), cov(for,against)=(var(total)-var(net))/4. They are not sixteen independent new measurements.',
        tree_representation_note='Adding columns changes randomized split views even with unchanged max_features and tree settings; score differences are not causal feature attribution.',
        source_of_checks='unchanged build_adjusted_rows exact identity, outcome and earlier forecast validation',
    )
    return result, coverage

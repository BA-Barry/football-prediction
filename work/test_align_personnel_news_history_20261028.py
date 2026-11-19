"""Focused failure cases for the descriptive personnel/news join."""
import json
import unittest

import align_personnel_news_history_20261028 as alignment


class AlignmentCases(unittest.TestCase):
    def test_reference_snapshot_at_its_forecast_cutoff_is_rejected(self):
        source = {'match_id': 'old', 'date': '2024-08-05', 'forecast_before_date': '2024-08-02',
            'teams': [{'team': 'A', 'source_team_code': 3, 'observed': True}],
            'archive_observation': {'path': 'old-source'}}
        snapshots = {'old-source': {'availability_proxy_at': '2024-08-02T00:00:00Z'}}
        with self.assertRaisesRegex(ValueError, 'strictly before historical F'):
            alignment.reference_source('old', 'A', '2024-08-20', {'old': source}, snapshots)

    def test_reference_match_on_target_cutoff_is_rejected(self):
        source = {'match_id': 'old', 'date': '2024-08-20', 'forecast_before_date': '2024-08-17'}
        with self.assertRaisesRegex(ValueError, 'strictly before the target F'):
            alignment.reference_source('old', 'A', '2024-08-20', {'old': source}, {})

    def test_same_player_code_at_another_team_is_not_a_state_pair(self):
        old = {'observation_id': 'other-team', 'snapshot_id': 'old',
               'source_team_code': 7, 'source_player_code': 42, 'news_id': 'n'}
        _, teams, codes, _ = alignment.build_indexes([old])
        current = {'source_team_code': 3, 'source_player_code': 42, 'news_id': 'n'}
        ref = {'snapshot_id': 'old', 'source_team_code': 3, 'observed': True}
        category, paired, candidates = alignment.classify_pair(current, ref, teams, codes, {})
        self.assertEqual(category, 'other_team_same_code_transfer_unknown')
        self.assertIsNone(paired)
        self.assertEqual(candidates, ['other-team'])

    def test_news_id_cannot_substitute_for_raw_text_equality(self):
        catalog = {'injured': {'raw_status': 'i', 'raw_news': 'Knee injury'},
                   'doubtful': {'raw_status': 'd', 'raw_news': 'Knee injury'},
                   'wording': {'raw_status': 'i', 'raw_news': 'Knee Injury'}}
        current = {'news_id': 'injured'}
        self.assertEqual(alignment.source_comparison(current, {'news_id': 'doubtful'}, catalog),
                         'same_news_status_changed')
        self.assertEqual(alignment.source_comparison(current, {'news_id': 'wording'}, catalog),
                         'same_status_news_changed')

    def test_duplicate_same_team_identity_is_explicitly_unpaired(self):
        rows = [{'observation_id': ident, 'snapshot_id': 'old', 'source_team_code': 3,
                 'source_player_code': 42, 'news_id': 'n'} for ident in ('first', 'second')]
        _, teams, codes, duplicates = alignment.build_indexes(rows)
        current = {'source_team_code': 3, 'source_player_code': 42, 'news_id': 'n'}
        ref = {'snapshot_id': 'old', 'source_team_code': 3, 'observed': True}
        category, paired, candidates = alignment.classify_pair(current, ref, teams, codes, {})
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(category, 'duplicate_reference_identity')
        self.assertIsNone(paired)
        self.assertEqual(candidates, ['first', 'second'])

    def test_projection_skips_labels_even_inside_nested_objects(self):
        selected = {'rows': {'*': {'match_id': True, 'fresh_feature_history': {'history': {
            'home': {'window_match_ids': {'5': True}}}}}}}
        value = {'rows': [{'match_id': 'm', 'actual_score': [4, 1], 'actual_outcome': 2,
            'unselected': {'escaped': 'value with \\" and } [ bracket', 'items': [1, 2]},
            'fresh_feature_history': {'history': {'home': {'window_match_ids': {'5': ['p1', 'p2'],
                '15': ['more']}, 'actual_outcome': 0}}}}]}
        first = alignment.project_json(json.dumps(value), selected)
        value['rows'][0].update(actual_score=[0, 2], actual_outcome=0)
        self.assertEqual(alignment.project_json(json.dumps(value), selected), first)
        self.assertEqual(first['rows'][0]['fresh_feature_history']['history']['home']['window_match_ids'],
                         {'5': ['p1', 'p2']})


if __name__ == '__main__':
    unittest.main()

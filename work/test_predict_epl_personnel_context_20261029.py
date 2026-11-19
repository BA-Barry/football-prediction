"""Ordinary checks of this current run, without fitting or scoring old forecasts."""
from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import joblib
import numpy as np

from predict_epl_personnel_context_20261029 import (
    FIT_DATE, MEASUREMENTS, OUT, SAVED_TREE, instant, side_facts,
)


class CurrentEPLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats=json.loads((OUT/'统计预测.json').read_text(encoding='utf-8'))
        cls.facts=json.loads((OUT/'给LLM的表现事实包.json').read_text(encoding='utf-8'))

    def test_all_ten_are_same_fixture_revisions(self):
        self.assertEqual(len(self.stats['records']),10)
        self.assertEqual(self.stats['new_unique_fixture_count'],0)
        self.assertEqual(self.stats['same_fixture_revision_count'],10)
        self.assertTrue(all(r['revision']['same_fixture_revision'] for r in self.stats['records']))
        self.assertTrue(all(instant(r['fixture']['kickoff'])>instant(self.stats['generated_at'])
                            for r in self.stats['records']))

    def test_two_fixed_fits_and_explicit_target_exclusion(self):
        m=self.stats['model_metadata']
        self.assertEqual(m['new_fits'],{'goal_simple':1,'independent_xg':1,'rich_121':0})
        target=set(m['target_exclusion']['excluded_target_ids'])
        for key in ('goal_parameters','xg_parameters'):
            p=m[key]
            self.assertEqual(p['regularization'],.01)
            self.assertEqual(p['half_life_days'],180)
            self.assertEqual(p['before_date'],FIT_DATE.isoformat())
            self.assertLess(p['training_max_date'],FIT_DATE.isoformat())
            self.assertEqual(len(p['reference_match_ids']),p['n_matches'])
            self.assertFalse(target & set(p['reference_match_ids']))
        self.assertEqual(self.stats['current_coverage']['completed_n'],80)
        self.assertEqual(self.stats['rich_source_report']['current_npxg_n'],80)
        self.assertEqual(self.stats['current_coverage']['overlapping_fd_n'],80)

    def test_probability_and_score_axes_agree(self):
        for r in self.stats['records']:
            for key in ('goal_simple','independent_xg','rich_121_research'):
                track=r[key]
                grid=np.asarray(track['score_grid']['probabilities'])
                h,a=np.indices(grid.shape)
                expected=[float(grid[h>a].sum()),float(grid[h==a].sum()),float(grid[h<a].sum())]
                self.assertTrue(np.isfinite(grid).all())
                self.assertTrue((grid>=0).all())
                self.assertAlmostEqual(float(grid.sum()),1,places=12)
                for actual,wanted in zip(track['probabilities'],expected,strict=True):
                    self.assertAlmostEqual(actual,wanted,places=12)
                self.assertAlmostEqual(sum(track['total_goals_0_to_7_plus'].values()),1,places=12)
                self.assertAlmostEqual(track['expected_total_goals'],sum(track['rates']),places=12)

    def test_saved_tree_and_base_models_reproduce(self):
        base=joblib.load(OUT/'本次英超统计模型.joblib')
        tree=joblib.load(SAVED_TREE)
        self.assertEqual(len(tree.feature_names),121)
        self.assertLess(tree.parameters()['training_max_date'],FIT_DATE.isoformat())
        pairs,dates=[],[]
        for r in self.stats['records']:
            d=instant(r['fixture']['kickoff']).date()
            pairs.append(r['rich_feature_details']['performance_features']);dates.append(d)
            for track,model in [('goal_simple','goal'),('independent_xg','xg')]:
                check=base[model].predict(r['home_model_identity'],r['away_model_identity'],'E0',d)
                for actual,wanted in zip(check,r[track]['rates'],strict=True):
                    self.assertAlmostEqual(actual,wanted,places=12)
        for r,check in zip(self.stats['records'],tree.predict_many(pairs,dates),strict=True):
            for actual,wanted in zip(check,r['rich_121_research']['rates'],strict=True):
                self.assertAlmostEqual(actual,wanted,places=12)

    def test_fact_packet_has_no_prediction_or_target_labels(self):
        forbidden={'rates','probabilities','expected_goals','expected_total_goals','forecast',
                   'odds','decimal_odds','actual_score','actual_outcome','full_time','llm','goal_simple',
                   'independent_xg','rich_121_research','log_goal_mean_for','log_xg_mean_for'}
        def inspect(value):
            if isinstance(value,dict):
                self.assertFalse(forbidden & set(value))
                for v in value.values():inspect(v)
            elif isinstance(value,list):
                for v in value:inspect(v)
        inspect(self.facts)

    def test_factual_windows_retain_actual_ids_dates_and_counts(self):
        for r in self.facts['records']:
            for side in r['sides'].values():
                for key in ('window_5','window_15'):
                    w=side[key]
                    self.assertEqual(w['n'],len(w['history']))
                    self.assertEqual(w['n'],min(w['requested_n'],side['history_count']))
                    self.assertEqual(len({q['match_id'] for q in w['history']}),w['n'])
                    self.assertTrue(all(q['date']<FIT_DATE.isoformat() for q in w['history']))
                self.assertEqual(side['window_5']['history'],side['window_15']['history'][-5:])

    def test_missing_measurement_is_unknown_without_window_substitution(self):
        values={name:1 for name in MEASUREMENTS}
        values['ppda_def']=None
        row=SimpleNamespace(match_id='synthetic-earlier-1',match_date=date(2026,10,28),
            team='synthetic-team',opponent='synthetic-opponent',competition='E0',is_home=True,**values)
        side=side_facts([row],'synthetic-team',True)
        self.assertEqual(side['window_15']['n'],1)
        self.assertIsNone(side['window_15']['measurements']['ppda_def'])
        self.assertIsNone(side['window_15']['measurements']['ppda_ratio'])
        self.assertEqual(side_facts([],'unknown-team',False)['identity_status'],'unknown')


if __name__=='__main__':
    unittest.main(verbosity=2)

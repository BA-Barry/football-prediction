"""Exercise the real entry flow with an advancing clock and lightweight models."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from football_model.all_competition_features import AllCompetitionFeatures
import football_model.current_projection as single_projection


WORK = Path(__file__).resolve().parents[2]/'work'
UTC = timezone.utc


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(WORK))
    spec = importlib.util.spec_from_file_location(
        'current_rich_entry_for_test', WORK/'run_current_rich_forecasts_20261026.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch(kickoff='2026-11-01T14:00:00+00:00'):
    fixture = {'match_id': 'timing-test:chelsea:arsenal', 'kickoff': kickoff}
    previous = {'records': [{'fixture': fixture, 'tracks': {
        'statistical': {'probabilities': [.4, .3, .3]}}}]}
    confirmation = {'checked_at': '2026-10-26T02:55:08.782562+00:00',
        'confirmed_unstarted_count': 1, 'fixture_observations': [{
            'original_fixture': fixture, 'confirmed_unstarted': True,
            'home_team_id': 1, 'away_team_id': 2}]}
    return previous, confirmation


def clock(monkeypatch, module, *times):
    moments = iter(times)

    class ControlledDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(moments)

    monkeypatch.setattr(module, 'datetime', ControlledDateTime)


def test_expired_rich_target_stops_before_any_source_loading_or_fit(entry, monkeypatch):
    old, confirmed = batch()
    clock(monkeypatch, entry, datetime(2026, 11, 4, tzinfo=UTC))
    monkeypatch.setattr(entry, 'read', lambda path: old if path == entry.OLD else confirmed)
    calls = []
    monkeypatch.setattr(entry, '_history', lambda *a: calls.append('history'))
    import refresh_current_all_competition_20261026 as schedule
    monkeypatch.setattr(schedule, 'load_current_schedule', lambda: calls.append('schedule'))
    import current_rich_model_helpers_20261026 as helpers
    monkeypatch.setattr(helpers, 'train_current_candidate', lambda *a, **kw: calls.append('fit'))
    with pytest.raises(ValueError, match='no longer pre-match'):
        entry.main()
    assert calls == []


def test_expired_single_projection_does_not_read_history_or_fit(monkeypatch, tmp_path):
    clock(monkeypatch, single_projection, datetime(2026, 11, 4, tzinfo=UTC))
    calls = []
    monkeypatch.setattr(single_projection, 'season_directories', lambda root: calls.append('history'))
    with pytest.raises(ValueError, match='after kickoff'):
        single_projection.run_projection(tmp_path, tmp_path)
    assert calls == []


def test_exact_kickoff_is_already_too_late_and_offsets_are_resolved(entry):
    old, _ = batch('2026-11-01T22:00:00+08:00')
    assert entry.validate_future_targets(old['records'], datetime(2026, 11, 1, 13, 59, tzinfo=UTC)) == [0]
    with pytest.raises(ValueError, match='no longer pre-match'):
        entry.validate_future_targets(old['records'], datetime(2026, 11, 1, 14, tzinfo=UTC))


def test_archived_confirmation_keeps_real_observation_time_without_expiry_rule(entry):
    old, confirmed = batch()
    now = datetime(2026, 11, 1, 13, 59, tzinfo=UTC)
    info = entry.archived_target_confirmation(old, confirmed, now)
    assert info['observed_at'] == confirmed['checked_at']
    assert info['age_seconds_at_start'] > 6*86400
    assert info['live_status_refetched'] is False
    assert info['publication_lead_days_at_start'] == [0]


@pytest.mark.parametrize('change', ['short_observation_list', 'future_observation'])
def test_incomplete_or_future_confirmation_is_not_accepted(entry, change):
    old, confirmed = batch()
    if change == 'short_observation_list':
        confirmed['fixture_observations'] = []
    else:
        confirmed['checked_at'] = '2026-11-01T15:00:00+00:00'
    with pytest.raises(ValueError):
        entry.archived_target_confirmation(old, confirmed, datetime(2026, 11, 1, 13, tzinfo=UTC))


def stub_main(entry, monkeypatch, tmp_path, generated_at):
    """Keep real orchestration, date feature composition, score conversion and IO."""
    old, confirmed = batch()
    clock(monkeypatch, entry, datetime(2026, 11, 1, 13, 59, 59, tzinfo=UTC), generated_at)
    monkeypatch.setattr(entry, 'ROOT', tmp_path)
    monkeypatch.setattr(entry, 'read', lambda path: old if path == entry.OLD else confirmed)
    monkeypatch.setattr(entry, '_history', lambda *a: ([], []))
    monkeypatch.setattr(entry, 'current_independent', lambda *a: (
        [], {1: 'Chelsea', 2: 'Arsenal'}, {'fd_overlapping_scores_checked': 0}))
    fits = []

    class Base:
        def __init__(self, *args):
            pass

        def fit(self, records, before):
            fits.append(('base', before))
            return self

        def predict(self, *args):
            return (1.4, 1.1)

        def parameters(self):
            return {}

    class Tree:
        def predict_many(self, pairs, dates):
            assert len(pairs) == len(dates) == 1
            assert all(side['forecast_log1p_lead_days'] == 0 for side in pairs[0])
            return [(1.3, 1.0)]

        def parameters(self):
            return {}

    def train(*args, **kwargs):
        fits.append(('tree', kwargs['before_date']))
        return Tree(), {'kind': 'test_stub', 'n_matches': 0, 'training_min_date': None,
            'training_max_date': None, 'source_feature_key': kwargs.get('feature_key', 'fresh_allcomp_features')}

    monkeypatch.setattr(entry, 'DateTeamGoalModel', Base)
    monkeypatch.setattr(entry, 'DateTeamXgModel', Base)
    import current_rich_model_helpers_20261026 as helpers
    monkeypatch.setattr(helpers, 'train_current_candidate', train)
    monkeypatch.setattr(helpers, 'load_current_rich_sources', lambda *a, **kw: {
        'np_records': {'example': SimpleNamespace(competition='E0')},
        'observations': [], 'reports': {}})
    import refresh_current_all_competition_20261026 as schedule
    monkeypatch.setattr(schedule, 'load_current_schedule', lambda: (
        AllCompetitionFeatures([]), {'directory': str(tmp_path)}))
    stored_models = {}
    monkeypatch.setattr(entry.joblib, 'dump', lambda model, path: stored_models.update({path: model}))
    monkeypatch.setattr(entry.joblib, 'load', lambda path: stored_models[path])
    saved_forecasts = {}
    monkeypatch.setattr(entry, 'write', lambda path, value: saved_forecasts.update({path.name: value}))
    return fits, saved_forecasts


def test_run_resumed_after_kickoff_cannot_save_prematch_probabilities(entry, monkeypatch, tmp_path):
    fits, saved = stub_main(entry, monkeypatch, tmp_path, datetime(2026, 11, 1, 14, tzinfo=UTC))
    with pytest.raises(ValueError, match='no longer pre-match'):
        entry.main()
    assert len(fits) == 4
    assert saved == {}


def test_unstarted_same_day_flow_uses_real_feature_path_and_preserves_snapshot_age(entry, monkeypatch, tmp_path):
    generated = datetime(2026, 11, 1, 13, 59, 59, 500000, tzinfo=UTC)
    fits, saved = stub_main(entry, monkeypatch, tmp_path, generated)
    entry.main()
    assert len(fits) == 4
    result = saved['完整预测与来源.json']
    assert result['generated_at'] == generated.isoformat()
    assert result['target_confirmation']['observed_at'].startswith('2026-10-26')
    assert result['target_confirmation']['live_status_refetched'] is False
    assert result['records'][0]['current_feature_details']['timing']['publication_lead_days'] == 0
    assert result['class_order'] == ['home', 'draw', 'away']
    assert '0-0 calendar-day' in result['limits'][1]
    assert saved['运行核验.json']['future_targets_checked_at'] == generated.isoformat()

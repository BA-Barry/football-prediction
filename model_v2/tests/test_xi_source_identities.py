"""Synthetic provider isolation, with no real data or parameter search."""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from football_model.hierarchical_xi_model import HierarchicalStartingXIModel
from football_model.starting_xi_model import (
    FullMatchStartingXIModel, UnknownStartingXIIdentityError, _fixture,
)


CUTOFF = date(2024, 1, 1)
MODELS = (FullMatchStartingXIModel, HierarchicalStartingXIModel)


def fixture(index=0, source="impect", day=None):
    roles = ["GK"] + ["DEF"] * 4 + ["MID"] * 3 + ["FWD"] * 3
    row = {"match_id": f"{source}:match:{index}",
           "match_date": (day or date(2023, 9, 1) + timedelta(days=index)).isoformat(),
           "home_team_id": 1, "away_team_id": 2,
           "full_time": [index % 3, (index + 1) % 2]}
    for side in ("home", "away"):
        players = [f"{source}:player:{side}{number}" for number in range(11)]
        if side == "home" and index % 2:
            players[-1] = f"{source}:player:rotation"
        row[f"{side}_starters"] = players
        row[f"{side}_roles"] = dict(zip(players, roles))
    return row


def history(source="impect"):
    return [fixture(index, source) for index in range(8)]


def predict(model, target):
    if isinstance(model, HierarchicalStartingXIModel):
        return model.predict_distribution(target)
    return model.predict_match(target)


def restore(model, payload):
    if isinstance(model, HierarchicalStartingXIModel):
        return type(model).from_parameters(payload, model.precision_cholesky)
    return type(model).from_parameters(payload)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("include_players", [False, True])
def test_provider_relabel_preserves_numerical_model_and_old_payload_structure(model_type, include_players):
    old = model_type(include_players=include_players).fit(history(), CUTOFF)
    sb = model_type(include_players=include_players).fit(history("statsbomb"), CUTOFF)
    assert sb.feature_names == [name.replace("impect:", "statsbomb:") for name in old.feature_names]
    assert np.array_equal(old.coefficients, sb.coefficients)
    assert np.array_equal(old.penalties, sb.penalties)
    assert old.optimizer_diagnostics == sb.optimizer_diagnostics
    if isinstance(old, HierarchicalStartingXIModel):
        assert np.array_equal(old.precision_cholesky, sb.precision_cholesky)
    for model, source in ((old, "impect"), (sb, "statsbomb")):
        payload = model.parameters()
        assert "identity_source" not in payload
        saved = restore(model, payload)
        assert saved.identity_source == source
        assert saved.parameters() == payload
        target = fixture(100, source, CUTOFF)
        assert predict(saved, target) == predict(model, target)
    old_prediction = predict(old, fixture(100, "impect", CUTOFF))
    sb_prediction = predict(sb, fixture(100, "statsbomb", CUTOFF))
    for key in ("log_mean", "log_cov", "map_means") if isinstance(old, HierarchicalStartingXIModel) else ("home_mean", "away_mean"):
        assert np.array_equal(old_prediction[key], sb_prediction[key])


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("include_players", [False, True])
def test_history_cannot_mix_providers_even_with_distinct_match_ids(model_type, include_players):
    rows = history() + [fixture(20, "statsbomb")]
    with pytest.raises(ValueError, match="one source namespace"):
        model_type(include_players=include_players).fit(rows, CUTOFF)


@pytest.mark.parametrize("mutation", ["match", "one_player", "opposing_roster"])
def test_a_fixture_match_and_every_starting_player_must_share_one_source(mutation):
    row = fixture()
    if mutation == "match":
        row["match_id"] = "statsbomb:match:0"
    elif mutation == "one_player":
        row["home_starters"][0] = "statsbomb:player:home0"
    else:
        row["away_starters"] = fixture(source="statsbomb")["away_starters"]
    with pytest.raises(ValueError, match="one source namespace"):
        _fixture(row)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("include_players", [False, True])
def test_prediction_cannot_reuse_same_numbered_team_from_another_provider(model_type, include_players):
    model = model_type(include_players=include_players).fit(history(), CUTOFF)
    with pytest.raises(ValueError, match="Prediction source namespace"):
        predict(model, fixture(100, "statsbomb", CUTOFF))


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("mutation", ["one_match", "all_matches", "one_player", "all_players", "reference_only"])
def test_restore_checks_existing_match_player_and_reference_source_identities(model_type, mutation):
    model = model_type().fit(history(), CUTOFF)
    payload = deepcopy(model.parameters())
    if mutation == "one_match":
        payload["reference_match_ids"][0] = "statsbomb:match:0"
    elif mutation == "all_matches":
        payload["reference_match_ids"] = [mid.replace("impect:", "statsbomb:") for mid in payload["reference_match_ids"]]
    elif mutation == "one_player":
        payload["players"][0] = payload["players"][0].replace("impect:", "statsbomb:")
    elif mutation == "all_players":
        payload["players"] = [pid.replace("impect:", "statsbomb:") for pid in payload["players"]]
    else:
        values = payload["reference_frequencies"]["1"]
        player = next(iter(values))
        values[player.replace("impect:", "statsbomb:")] = values.pop(player)
    with pytest.raises(ValueError, match="source namespace|reference frequencies"):
        restore(model, payload)


@pytest.mark.parametrize("model_type", MODELS)
def test_future_foreign_or_invalid_identity_content_is_not_read(model_type):
    rows = history("statsbomb")
    model = model_type().fit(rows, CUTOFF)
    future = [{"match_date": CUTOFF.isoformat(), "match_id": "impect:match:future",
               "home_starters": object(), "home_roles": object(), "full_time": object()},
              {"match_date": "2024-01-02", "match_id": "unsupported", "home_team_id": False}]
    repeated = model_type().fit(rows + future, CUTOFF)
    assert repeated.parameters() == model.parameters()
    if isinstance(model, HierarchicalStartingXIModel):
        assert np.array_equal(repeated.precision_cholesky, model.precision_cholesky)


@pytest.mark.parametrize("source", ["impect", "statsbomb"])
def test_unknown_player_policies_remain_distinct_and_unchanged(source):
    target = fixture(100, source, CUTOFF)
    old_player = target["home_starters"][-1]
    new_player = f"{source}:player:unseen"
    target["home_starters"][-1] = new_player
    target["home_roles"][new_player] = target["home_roles"].pop(old_player)
    fixed = FullMatchStartingXIModel().fit(history(source), CUTOFF)
    with pytest.raises(UnknownStartingXIIdentityError, match="Unknown starting players"):
        fixed.predict_match(target)
    hierarchical = HierarchicalStartingXIModel().fit(history(source), CUTOFF)
    prediction = hierarchical.predict_distribution(target)
    assert prediction["unknown_player_ids"] == [new_player]
    assert np.asarray(prediction["unknown_player_log_cov"]) == pytest.approx(np.eye(2) * .005)


@pytest.mark.parametrize("model_type", MODELS)
def test_statsbomb_duplicate_matches_are_not_a_new_training_sample(model_type):
    rows = history("statsbomb")
    with pytest.raises(ValueError, match="Duplicate match IDs"):
        model_type().fit(rows + [deepcopy(rows[0])], CUTOFF)


@pytest.mark.parametrize("source", ["other", "", "StatsBomb"])
def test_unrecognized_provider_is_not_silently_relabelled(source):
    with pytest.raises(ValueError, match="namespaced"):
        _fixture(fixture(source=source))

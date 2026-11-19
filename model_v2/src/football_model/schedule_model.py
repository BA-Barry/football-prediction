"""Research-only Poisson extension using recorded source fixture dates.

The source is not a complete calendar: cups, other leagues, international duty,
and publication timestamps are not reconstructed. Counts describe observed
same-competition source records, not real workload or verified rest. Excluded
goal-fitting records still contribute dates; their original date semantics are
retained. All fixtures on the queried calendar date are excluded together.
"""

from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta
import math
from numbers import Integral
from typing import Sequence, TYPE_CHECKING

from .date_model import DateTeamGoalModel
from .team_model import fit_weighted_poisson, team_features

if TYPE_CHECKING:
    from .football_data import HistoricalMatch


WINDOW_DAYS = 14
ACTIVITY_SCOPE = "source_recorded_same_competition_fixture_dates"
SELF_COUNT_FEATURE = "context:self_fixture_count_prev14d"
OPPONENT_COUNT_FEATURE = "context:opponent_fixture_count_prev14d"
MODEL_TYPE = "date_level_source_schedule_regularized_poisson"
SCHEDULE_FEATURE_DEFINITION = {
    "scope": ACTIVITY_SCOPE,
    "feature_names": [SELF_COUNT_FEATURE, OPPONENT_COUNT_FEATURE],
    "window_days": WINDOW_DAYS,
    "window": "[match_date-14_days,match_date)",
    "unit": "source_fixture_records",
    "numeric_transform": "none",
    "includes_goal_fit_excluded_source_records": True,
    "same_date_records_included": False,
    "result_or_market_fields_read_by_date_index": False,
    "date_context_updates_at_each_prediction_date": True,
    "coefficients_fixed_between_explicit_fits": True,
    "rest_gap_used_as_model_feature": False,
    "coverage_metadata_used_as_model_feature": False,
    "missing_other_competitions_or_sources_imputed": False,
    "zero_count_meaning": "no_same_competition_source_fixture_records_in_window",
    "special_date_semantics": "original_source_dates_retained_not_verified_full_match_load",
    "not_measured": "complete_workload_true_rest_or_fixture_publication_availability",
}


def _identity(home, away, competition):
    if not all(isinstance(value, str) and value.strip()
               for value in (home, away, competition)):
        raise ValueError("Team and competition identifiers must be nonempty strings")
    if home == away:
        raise ValueError("Teams must differ")


class DateActivityIndex:
    """Index identity/date/exclusion metadata only, without reading score/odds.

    Team identifiers remain verbatim (including their source namespaces).
    Competition is a separate index key, so dates in another league cannot
    contribute even when supplied a matching team string.
    """

    def __init__(self, records: Sequence["HistoricalMatch"]):
        activity = defaultdict(list)
        first_dates = {}
        seen_ids = set()
        for record in records:
            # Deliberately do not copy vars(record), full_time, or markets.
            match_id, competition = record.match_id, record.competition
            played_date, home, away = record.match_date, record.home, record.away
            exclusion = record.goal_fit_exclusion
            if not isinstance(match_id, str) or not match_id.strip():
                raise ValueError("Source match identifiers must be nonempty strings")
            if match_id in seen_ids:
                raise ValueError("Duplicate match identifiers in source date records")
            seen_ids.add(match_id)
            _identity(home, away, competition)
            if type(played_date) is not date:
                raise ValueError("Source fixture dates must be explicit calendar dates")
            if exclusion is not None and (not isinstance(exclusion, str) or not exclusion.strip()):
                raise ValueError("Source exclusion metadata must be null or a nonempty string")
            first_dates[competition] = min(played_date, first_dates.get(competition, played_date))
            for team in (home, away):
                activity[(competition, team)].append((played_date, match_id, exclusion is not None))
        self._activity = {key: tuple(sorted(values)) for key, values in activity.items()}
        self._dates = {key: tuple(value[0] for value in values)
                       for key, values in self._activity.items()}
        self._first_dates = first_dates

    def context(self, home: str, away: str, competition: str, on_date: date):
        _identity(home, away, competition)
        if type(on_date) is not date:
            raise ValueError("Query with an explicit calendar date")
        try:
            window_start = on_date - timedelta(days=WINDOW_DAYS)
        except OverflowError as error:
            raise ValueError("Query date cannot represent the previous 14-day window") from error
        first_date = self._first_dates.get(competition)
        left_censored = first_date is None or first_date > window_start

        def side(team):
            key = (competition, team)
            values, dates = self._activity.get(key, ()), self._dates.get(key, ())
            left, right = bisect_left(dates, window_start), bisect_left(dates, on_date)
            recent = values[left:right]
            previous_date = dates[right - 1] if right else None
            return {
                "team": team,
                "source_fixture_count_prev14d": right - left,
                "previous_date": previous_date.isoformat() if previous_date is not None else None,
                "gap_days": (on_date - previous_date).days if previous_date is not None else None,
                "previous_14d_ids": [value[1] for value in recent],
                "special_previous_14d_ids": [value[1] for value in recent if value[2]],
                "window_left_censored": left_censored,
                "has_previous_source_record": previous_date is not None,
            }

        return {
            "home": side(home), "away": side(away), "competition": competition,
            "on_date": on_date.isoformat(), "window_days": WINDOW_DAYS,
            "scope": ACTIVITY_SCOPE,
        }


def source_date_features(home: str, away: str, competition: str, context):
    """Original team features plus only own/opponent observed fixture counts."""
    _identity(home, away, competition)
    if not isinstance(context, dict) or context.get("competition") != competition:
        raise ValueError("Date context must match the requested competition")
    if context.get("scope") != ACTIVITY_SCOPE or context.get("window_days") != WINDOW_DAYS:
        raise ValueError("Unexpected source fixture date scope or window")
    counts = []
    for side, team in (("home", home), ("away", away)):
        values = context.get(side)
        if not isinstance(values, dict) or values.get("team") != team:
            raise ValueError("Date context must match the requested teams and sides")
        count = values.get("source_fixture_count_prev14d")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError("Source fixture counts must be nonnegative integers")
        counts.append(int(count))
    features = team_features(home, away, competition)
    for side, own_count, opponent_count in ((0, counts[0], counts[1]), (1, counts[1], counts[0])):
        features[side][SELF_COUNT_FEATURE] = own_count
        features[side][OPPONENT_COUNT_FEATURE] = opponent_count
    return features


class DateScheduleGoalModel(DateTeamGoalModel):
    """Non-adopted date-level research arm; does not alter the original model."""

    def fit(self, records: Sequence["HistoricalMatch"], before_date: date):
        if type(before_date) is not date:
            raise ValueError("Use an explicit calendar date, not a fabricated timestamp")
        records = list(records)
        activity_index = DateActivityIndex(records)
        history = sorted((record for record in records if record.match_date < before_date
                          and record.goal_fit_exclusion is None),
                         key=lambda record: (record.match_date, record.match_id))
        if not history:
            raise ValueError("No eligible earlier-date results")
        features, target, weights = [], [], []
        for record in history:
            score = record.full_time
            if not isinstance(score, (tuple, list)) or len(score) != 2 or any(
                    isinstance(goal, bool) or not isinstance(goal, Integral) or goal < 0
                    for goal in score):
                raise ValueError("Training scores must contain two nonnegative integer goals")
            context = activity_index.context(record.home, record.away, record.competition,
                                             record.match_date)
            features.extend(source_date_features(record.home, record.away, record.competition,
                                                 context))
            target.extend(score)
            weight = math.exp(-math.log(2) * (before_date - record.match_date).days /
                              self.half_life_days)
            weights.extend((weight, weight))
        vectorizer, model = fit_weighted_poisson(features, target, weights, self.regularization)
        self.coefficients = dict(zip(vectorizer.get_feature_names_out(), map(float, model.coef_)))
        self.intercept = float(model.intercept_)
        self.before_date = before_date
        self.teams = {team for record in history for team in (record.home, record.away)}
        self.competitions = {record.competition for record in history}
        self.n_matches = len(history)
        self.training_min_date = history[0].match_date
        self.training_max_date = history[-1].match_date
        self.reference_match_ids = [record.match_id for record in history]
        self.optimizer_iterations = int(model.n_iter_)
        self.activity_index = activity_index
        return self

    def predict(self, home: str, away: str, competition: str, match_date: date):
        if not hasattr(self, "coefficients"):
            raise ValueError("Fit earlier-date results before predicting")
        if type(match_date) is not date or match_date < self.before_date:
            raise ValueError("Prediction must not precede the fit cutoff date")
        _identity(home, away, competition)
        if {home, away} - self.teams:
            raise ValueError("Unseen teams require an explicitly estimated prior")
        if competition not in self.competitions:
            raise ValueError("Unseen competition")
        context = self.activity_index.context(home, away, competition, match_date)
        features = source_date_features(home, away, competition, context)
        return tuple(math.exp(self.intercept + sum(self.coefficients.get(key, 0) * value
                     for key, value in row.items())) for row in features)

    def parameters(self):
        parameters = super().parameters()
        parameters["model_type"] = MODEL_TYPE
        parameters["schedule_feature_definition"] = deepcopy(SCHEDULE_FEATURE_DEFINITION)
        return parameters

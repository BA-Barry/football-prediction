# Multi-rating feature adapter

`football_model.multi_rating.build_rating_features` is an in-memory adaptation of the Apache-2.0 [Spoho football sparsity project](https://github.com/spoho-datascience/sparsity-sports-forecasting). Its Elo and pi equations/constants follow `RatingModels.py` and `ELOXGBoostModel.py`. It neither imports nor executes upstream scripts with machine-specific file writes or missing database modules.

Observations provide `date`, `league`, `home_team`, `away_team`, `home_goals`, `away_goals`, plus optional `home_shots`/`away_shots` and `home_shots_on_target`/`away_shots_on_target`. Targets provide `match_id`, `forecast_date`, `league` and team identifiers. Dates are calendar dates or ISO strings. A match must occur once in observations. Identities are isolated by league and stable team identifier; caller handles source mappings.

Only observations dated strictly before each forecast date enter that target's ratings. A later target can use newly observed matches without changing earlier targets. Internal zero priors are not reported as observed zero strength: rating outputs remain `None` until the corresponding history exists; features also include observation/update counts and ages. Missing shot data carry the last known rating forward with explicit coverage.

The source transforms observed shots into result probabilities with a two-input XGBoost classifier before updating shot Elo. Here the classifier uses only matches before each July 1 and is frozen through the following June. It retains 100 trees, depth 6, learning rate 0.3, with explicit seed and one CPU thread. This replaces the original random two-fold step, which would let later match outcomes influence historical features. If any outcome class is absent, that period's adjustment is unknown rather than invented. Actual fits and exact training date bounds are returned in `metadata.shot_adjustment_splits`.

There is no target-market rating because historical quote availability times are unverified, and no starting-lineup market value input because the user's prediction does not wait for target starting lineups. Promotion/relegation future backfills are omitted; unknown team zero priors and no learned cross-league prior are documented limitations. This is a time-adapted research component, not an exact reproduction of the full paper and not a replacement of current forecasts.

The license is in `licenses/sparsity-sports-forecasting.txt`. New source changes date: 2026-10-26. Upstream code's author label is Dell-PC; repository organization is spoho-datascience. See the original repository for author and paper context.

## Full candidate execution

`work/run_multi_rating_joint_20261026.ps1` configures the G-drive runtime and runs the complete study. The preparer `work/prepare_multi_rating_observations_20261026.py` uses only existing, exactly matched historical files. The runner has `--prepare-only` and `--reuse-prepared` options; invoking the full run fits the two declared candidates rather than silently selecting a winner.

The classifier `football_model.multi_rating_xgb.FixedMultiRatingXGB` uses local H=0/D=1/A=2 coding, native NaN and explicit modern defaults (100 rounds, depth 6, learning rate 0.3, uniform fixture weights). Its output is 1X2 only. The core design has 43 inputs; the full joint design has 236, including existing dated performance, schedules and EPL availability descriptions. Current odds, target lineup values and LLM judgments are not silently substituted into missing fields.

The first full study completed 44 main classifier fits and 10 chronological shot-adjustment fits. Two subsequent temperatures reused earlier out-of-sample 2024/25 predictions, were fixed before 2025/26, and did not alter the predicted top classes. Both new candidates remain below the original system on the complete later-season cohort. They remain reusable research components with measured complementary errors; this does not establish either universal inferiority or deployable ensemble improvement.

Results and all saved models: [study report](G:/Codex/football-prediction-2026-10-22/outputs/多路评级联合接入_20261026/本轮完成与下一步.md). XGBoost 3.4.1 was installed in the G-drive `.venv`; the optional dependency group is `multi-rating`.

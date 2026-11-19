# Reused code and dependencies

`src/football_model/vendor/penaltyblog_grid.py` is an unmodified copy of
`penaltyblog/models/football_probability_grid.py` from
[martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog),
default-branch revision dated 2026-10-19, reviewed 2026-10-22. The upstream
copyright and MIT permission text are preserved in
`src/football_model/vendor/PENALTYBLOG_LICENCE.txt`.

The original narrow-reuse path imports only `FootballProbabilityGrid`. The upstream
`create_dixon_coles_grid` factory in the reviewed file is deliberately not used:
the off-diagonal low-score factors disagree with the same library's fitted
model in `models/probabilities.pyx`. Our `probability.score_grid` implements the
standard factors and is checked with asymmetric goal rates. This observation
does not imply that penaltyblog's fitted Dixon-Coles model has that error.

Full-package update (v0.5.0, 2026-10-22): penaltyblog 1.13.1 (MIT) and its
normal dependencies are now installed in the G-drive virtual environment.
An earlier binary-only attempt stopped at `socks`; the subsequent normal
installation built that dependency successfully. No upstream package files
were patched. `penaltyblog_adapter.py` now uses native `PoissonGoalsModel`,
`DixonColesGoalModel`, `dixon_coles_weights`, native prediction and persistence.
This fitted-model path does not call the external-lambda factory discussed
above. Other packaged models are available but have not been trained here.

Runtime dependencies installed in the G-drive environment:

- NumPy (BSD-3-Clause), from the existing shared interpreter environment.
- SciPy 1.18.1 (BSD-3-Clause), G-drive virtual environment.
- scikit-learn 1.9.1 (BSD-3-Clause), G-drive virtual environment; directly reused
  `DictVectorizer` and `PoissonRegressor` for the trainable team component.
- penaltyblog 1.13.1 (MIT), G-drive virtual environment; native parameter
  estimation and score-distribution predictions in the four-season experiment.
- pytest 9.1.1 (MIT), G-drive virtual environment, for ordinary unit and
  integration tests, including native fit, prediction, persistence and temporal selection.

Other GitHub projects in the research report were inspected for methods and
selection only. Their code and datasets are not incorporated into this package.
In particular, projects without an explicit reuse license and footymodel's
all-rights-reserved code are not copied. Data-source terms remain separate
from a reader library's software license.

Data integration history (2026-10-22): five 2024/25 match-result JSON files and
their repository-declared CC0 license were fetched from openfootball/football.json
into the G-drive data directory, outside this source package. This is a field
inspection, not model training; per-match original provenance is not fully
verified. No openfootball source code was copied.

Football-Data.co.uk initially was not downloaded following its stated use
restrictions. The user subsequently explicitly requested official downloads
for a personal, noncommercial football statistics/entertainment model, not
foundation-model training or an external product. Five official 2024/25 CSVs
were then downloaded and used in v0.4. In v0.5 another 15 official CSVs cover
2022/23, 2023/24 and 2025/26. Source/use records and untouched raw files remain
on G:, outside the source distribution. No mirror/access-control workaround
or public data redistribution is used. The user's stated purpose is not an
additional license grant; data terms remain distinct from this software's MIT
dependencies. This provenance record does not claim timestamp-complete data
or complete detection of every historical special match.

Player-event integration (v0.6.0): official StatsBomb Open Data now redirects
from `statsbomb/open-data` to `hudl/open-data`. We downloaded the 34 published
Bundesliga 2023/24 match event/lineup pairs for an internal, personal,
noncommercial parser and provenance study, with original JSON, README and the
Public Data User Agreement retained under G-drive data/. The competition-season
catalog does not imply complete coverage: all 34 matches feature Leverkusen.
`statsbomb_stints.py` and `player_probe.py` are original code, not copied reader
code. Raw/derived data are outside the source package and not redistributed.
StatsBomb's data agreement is not CC0 or MIT; no commercial/public service is
provided. Its attribution/branding requirements apply to published analyses;
its request to express research interest has not been fulfilled on the user's
behalf, and no contact details/account registration have been submitted.

Full-season player experiment (v0.7.0): the same official Hudl/StatsBomb source
provides all 380 Premier League 2015/16 event/lineup pairs, downloaded with
source agreement and per-file retrieval records to G-drive data/. The complete
home/away schedule was measured from metadata. Another official Football-Data
2015/16 E0.csv is retained outside the source distribution for independent
date/team/full-time-score reconciliation only; its odds are not used in this
experiment. Original/derived observations are not included in the source ZIP.
No new third-party source code was copied. The original joint player/team
point-process implementation directly uses SciPy sparse matrices/L-BFGS-B and
NumPy, already installed dependencies. Historical availability/lineup clocks
remain unknown; retrospective results are not advertised as pre-match scores.

Second-season data integration (v0.8.0): the official Hudl/StatsBomb catalog's
listed men's top-five league match files were actually inspected, not assumed
complete from their labels. No second complete later EPL season was available
in that catalog. Following the user's explicit choice, the original
Pappalardo/Massucco Wyscout public 2017/18 England files were downloaded directly
from Figshare. The retained article metadata declares CC BY 4.0. The authors,
original archives, selected England members and retrieval records remain under
G-drive data/wyscout/, outside the source distribution. Attribution:
Pappalardo, Luca; Massucco, Emanuele (2019), Events,
https://doi.org/10.6084/m9.figshare.7770599.v1, and Matches,
https://doi.org/10.6084/m9.figshare.7770422.v1. Related paper:
Pappalardo et al., A public data set of spatio-temporal match events in soccer
competitions, Scientific Data (2019), DOI 10.1038/s41597-019-0247-7.
The retained metadata is authoritative for each file's precise citation/version.

An additional official Football-Data 2017/18 E0.csv is used only for independent
date/home/away/FT/HT/red-card reconciliation here, not odds or fitting. No
original or derived observations are included in the source ZIP. The new
streaming reader and diagnostic probe are original code. The socceraction
project's public Wyscout reader was consulted for field and minute concepts;
no source code was copied and that package was not installed or downgraded.
Its estimated half duration is not treated as a true whistle timestamp.
The user then chose an earlier-training-only minute-precision degradation
experiment on already retained StatsBomb data; this neither generates Wyscout
xG nor repairs its original goals/cards/clocks. Cross-provider numeric player
IDs are not merged. Timing sensitivity does not establish a new-season
forecast improvement or current betting value.

Second-season experiments (v0.9.0): following the user's explicit approval,
the already retained official Football-Data 2015/16 and 2017/18 CSVs now support
separate complete-score weekly team experiments and their B365 pre-closing
de-vig comparison. The earlier paragraph saying "not odds or fitting" describes
the v0.8.0 diagnostic task only. This does not provide a new license grant or
verified historical quote-availability timestamp. Source CSVs remain untouched
and outside the source ZIP; no ROI or current betting service is provided.

The Wyscout files also support an explicitly approximate two-fixed-half-window,
goal-only player study. All three declared minute-clock hypotheses are retained,
not selected using held-out scores. An absent source goal is not repaired;
stoppage goals outside the target still update relevant later state. Data and
derived observations are not included in the software source ZIP. The new
fixed-window adapter and experiment orchestration are original code. No
socceraction source code was copied, and its minute-expansion convention is not
claimed to identify the public files' actual substitution seconds. Attribution
to the original data authors and article versions remains as above.

Same-provider window/clock diagnostics (v0.10.0): only already retained official
Hudl/StatsBomb 2015/16 events/lineups and Football-Data score reconciliation are
used. No new downloads, dependencies or copied third-party code were added.
The original goal-only replay, SciPy joint model and chronological split helpers
are reused with an original deterministic goal-ID/clock-based window crop and
same-cohort summaries. True half ends shorter than the chosen window reject
the paired fixed-window case rather than extending recorded exposure. Full H1
stoppage state remains in H2. Approximate clock paths do not become actual
played minutes, and no new xG or pre-match probabilities are manufactured.
All source/derived observations, fitted models and raw data remain outside the
software source ZIP on G:. Provider licenses, attribution and the distinction
between personal statistical research and third-party permission remain as
recorded above.

Player regularization studies (v0.11.0): reuse only the same local fixed-half
StatsBomb and Wyscout research paths with original chronological selection,
same-cohort loss summaries, and existing SciPy L2 fitting code. No new download,
dependency, third-party source copying, market blending, unseen-player default,
or retrospective availability timestamp was added. Static validation and the
subsequently user-approved weekly forward validation are separate experiments;
both seasons were already viewed and neither is presented as a new blind test.
Very strong shrinkage does not grant permission to invent player effects or
extend the original providers' usage terms. All observations, saved parameters
and experimental output stay outside the distributable software source ZIP.

Player-only exposure scaling (v0.12.0): only the same local fixed-half research
paths and original SciPy joint model are reused. The player L2 multiplier is
computed from physical training-match clock exposure, referenced to the first
validation week's strictly earlier history; it is not inferred from held-out
goals or an imported third-party prior. Team/context penalties remain fixed.
The new exposure-selection runner and six-track arithmetic summaries are
original code; no dependency, external source, or new download was introduced.
Existing data licenses, attribution and personal-statistical-research limits
remain unchanged. No raw/derived observations or fitted parameters are included
in the software source ZIP; all working files and reports remain on G:.

Saved pre-match forecast diagnostics (v0.13.0): the new original NumPy runner
reads the existing native-model experiment and proportional de-vig probabilities;
it neither fits nor modifies models. Method references were checked against
[arch's non-circular moving block definition](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html)
and [scikit-learn's calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).
No upstream bootstrap source was copied and arch was not installed. Same-week
leagues are sampled jointly, empty calendar weeks are retained, and ratio-of-sums
weighting is explicit. Intervals are exploratory and conditional on saved forecasts,
not evidence of tradable market timing, individual-match certainty or new blind
validation. No new dependency, source data download or redistribution was added.

Earlier-only calibration research (v0.14.0): original NumPy/SciPy code estimates
one inverse temperature or two class offsets on 2024/25 out-of-time forecasts of
the unchanged original model. It uses the local three earlier Football-Data
seasons and existing 2025/26 saved forecasts, without new source downloads or
redistribution. Method references include [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html)
and scikit-learn calibration documentation; the paper's image/text findings do
not establish effectiveness for football. No upstream code was copied. Neither
CalibratedClassifierCV's default non-temporal folds nor a full-season-selected
native configuration is passed off as a chronological historical choice.
Calibrated 1X2 research vectors are explicitly separate from the unchanged
score-grid, total-goal and handicap predictions; there is no production adoption
or market fusion. All existing data terms and G-drive location constraints remain.

Fixed-anchor refresh research (v0.15.0): original code reuses the existing local
Football-Data four-season CSVs, source exclusions, and scikit-learn regularized
Poisson estimator. Only the calendar refresh interval varies (7/14/28 days);
the decay, regularization, proportional de-vig, and strict unknown-team policy
remain unchanged. No upstream source was copied, dependency installed or new
data downloaded. Two already-viewed seasons are reported as exploratory paired
research, not another blind validation or proof of tradable value. Full-time
1X2, five total-goal bins, and three-way away-minus-two probabilities derive from
each arm's own common score distribution; no independent calibration vector is
mixed into it. Independent arithmetic reuses the previous independent audit's
CSV reader, Poisson recurrence and explicit week-block equations by file path,
without importing the production model or repeating its optimizations. Raw and
derived observations, fitted parameters and audit results remain outside the
software source ZIP, on G:. Existing provider terms and attribution still apply.

Conditional goal-rate diagnostics (v0.16.0): original code reads only the local
completed refresh experiment and original CSV histories. It reconstructs each
rate from readable team coefficients and computes untruncated conditional
Poisson diagnostics, rather than interpreting a score-grid expectation as shot
xG or unconditional goal variance as conditional dispersion. Method references
were checked against [SciPy Poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html),
[scaled Bessel i0e](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.i0e.html),
and the non-circular moving-block documentation linked above. No upstream source
was copied and no new package or data was downloaded. The independent arithmetic
module imports no production-model code and adapts the earlier independent
audit's explicit calendar-week resampling equation; its count is not an enlarged
sample or optimizer re-fit. All three refresh arms and both already-viewed
seasons remain exploratory, with no calibration, market fusion, new player
prior or formal model replacement. Source ZIPs contain only code/tests/docs;
observations, fitted parameters, diagnostic outputs and audit results remain
outside them, on G:. Existing provider terms and attribution remain unchanged.

Earlier-fitted rate calibration (v0.17.0): original closed-form estimators use
only the existing local 2024/25 time-out predictions and actual goals. Shared
and home/away scales minimize the equal-match independent-Poisson score NLL;
2025/26 application rebuilds coherent score distributions without changing
the original forecast files. The experiment and independent arithmetic audit
reuse the existing explicit calendar-block research utilities. No external
source code, package, model weights or data was added. All methods remain
exploratory and unadopted; neither neural networks nor reinforcement learning
were trained or connected. Source ZIPs exclude data, fitted parameters and
research outputs. Existing source attribution and provider terms still apply.

Source fixture-date features (v0.18.0): original research code reuses the
existing scikit-learn PoissonRegressor and original local Football-Data CSVs.
Only two own/opponent counts of source-league records in [date-14,date) enter
the new GLM; no cup, international, injury, lineup or motivation data is
inferred. Historical excluded goal labels remain excluded, while their source
dates retain their original semantics in the date-only index. No new upstream
code, packages, weights or data were downloaded. Standalone audit arithmetic
reuses the adjacent independent probability and calendar-block modules and
adds handwritten source-count/feature/penalized-gradient equations. This is not
a refit, provider-truth audit or formal model adoption. All code, fitted models,
outputs and caches stay on G:; source ZIPs exclude data/parameters/predictions.
Existing Football-Data source attribution and personal-use conditions remain.

Continuous latent team-strength 1X2 research (2026-10-25) directly reuses
[KickScore 0.2.0](https://github.com/lucasmaystre/kickscore), Copyright (c) 2017
Lucas Maystre, under the MIT license. The unmodified installed package and its
MIT license are in the G: project virtual environment; the study also includes
a copy of that license. KickScore supplies Constant and Exponential Gaussian
processes and expectation-propagation inference. The local adapter provides
competition-specific football roles, strictly earlier date-prefix inputs, and
mathematically equivalent stable ordered-probit probability arithmetic. It does
not replicate the upstream paper's results or assert its performance transfers
to this dataset. New installation files and numba/pip caches are on G:.

GAP outcome research (2026-10-25) directly reuses unmodified statsmodels 0.15.0
`OrderedModel` for ordered-logit maximum-likelihood estimation and native
probability prediction. statsmodels and newly required patsy/formulaic and
supporting packages are installed in the G: virtual environment. Their license
texts are copied to `outputs/GAP胜平负研究_20261025/upstream/`; package versions
and installed locations are recorded in that study's `复用组件.json`.
`gap_model.py` locally implements the published GAP state equations from
Edward Wheatcroft (2021), DOI 10.3233/JSA-200462; no source from the reviewed
klynch999 gist or pratikkk97 R project is copied. The local ML mapping,
unweighted fixtures and omitted home-six-game burn-in differ from the paper.
The study is not a reproduction of its reported accuracy or gambling profits,
and its fitted artifacts do not replace existing forecasts.

Historical RDS archive import (2026-10-27) uses unmodified
[pyreadr 0.5.8](https://github.com/ofajardo/pyreadr), installed only in the G:
project virtual environment. Its installed distribution declares
AGPL-3.0-or-later and retains LICENSE plus the bundled librdata/readstat notices
under `pyreadr-0.5.8.dist-info/licenses/`. The optional `archive-data` extra
records this reader version; it is an import tool, not a forecasting algorithm.
Public Understat archive access and underlying data rights are separate from
the reader's software license. The source archive is retained with its own
origin and coverage record; no open-data license is inferred from this package.

The historical context reconstruction (2026-10-28) also inspects the author's
[Stage3 R transformation](https://github.com/UsDAnDreS/JQAS_LeveragingMinuteByMinute_Soccer_Event_Data_Paper/blob/main/Stage3_JQAS_MBM_Data_Wrangling_With_RedCards.R).
A read-only source snapshot is retained under the G: project's
`work/research/game_context_20261027/`; it is not executed or imported.
The new Python event-count bounds are locally implemented; they do not assume
that the upstream inferred period labels or mixed pre/post-state columns are
official event timing. Source snapshot provenance accompanies that file.

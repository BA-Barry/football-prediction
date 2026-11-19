param([ValidateSet('demo','test','inspect-data','openfootball-snapshot','market-research','market-fusion-research','laplace-diagnostics','season-state-research','minutes-research','core-tuning','score-shape','cmp-research','cmp-forward','dependence-forward','rate-calibration-forward','snapshot-dynamic','prematch-pilot','multiline-pilot','batch-pilot','batch-score','english-tiers-research','english-tiers-weighting','player-context','prospective-revision','impect-pilot','impect-season','impect-player-research','starting-xi-data','starting-xi-research','starting-xi-diagnostics','starting-roles-data','statsbomb-starting-roles','starting-xi-transfer','hierarchical-xi-research','backtest','native-train','forecast-diagnostics','calibration-research','refresh-research','goal-rate-diagnostics','rate-calibration-research','schedule-research','player-data','player-train','player-rolling','catalog-data','wyscout-data','wyscout-inspect','timing-measure','timing-fit','second-team','second-players','window-clock','player-shrinkage','player-forward-selection','player-exposure-scaling')][string]$Action = 'demo', [string]$BatchInput = '', [string[]]$ResultEvidence = @(), [string]$TierInput = '', [string]$TierReference = '', [string]$NewsInput = '', [string]$SnapshotInput = '', [string]$RevisionInput = '', [string]$SampleInput = '', [string]$SeasonInput = '', [string]$ParsedInput = '', [string]$InspectionInput = '', [string]$ResearchInput = '')
$ErrorActionPreference = 'Stop'
$taskRoot = if ([string]::IsNullOrWhiteSpace($env:FOOTBALL_MODEL_ROOT)) {
    Split-Path -Parent $PSScriptRoot
} else {
    [System.IO.Path]::GetFullPath($env:FOOTBALL_MODEL_ROOT)
}
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $taskRoot '.venv\bin\python'
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw "Virtual environment Python was not found. Run the setup command in README.md first."
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$env:TEMP = Join-Path $taskRoot 'work\tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $taskRoot 'work\pip-cache'
$env:MPLCONFIGDIR = Join-Path $taskRoot 'work\matplotlib'
$env:SOCCERDATA_DIR = Join-Path $taskRoot 'data\soccerdata'
$env:JOBLIB_TEMP_FOLDER = $env:TEMP
$env:NUMBA_CACHE_DIR = Join-Path $taskRoot 'work\numba-cache'
$env:IPYTHONDIR = Join-Path $taskRoot 'work\ipython'
$env:JUPYTER_CONFIG_DIR = Join-Path $taskRoot 'work\jupyter'
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location $PSScriptRoot
try {
    if ($Action -eq 'demo') { & $taskPython -m football_model.demo }
    elseif ($Action -eq 'inspect-data') { & $taskPython -m football_model.data_probe }
    elseif ($Action -eq 'openfootball-snapshot') { & $taskPython -m football_model.openfootball_snapshot }
    elseif ($Action -eq 'market-research') { & $taskPython -m football_model.market_research }
    elseif ($Action -eq 'market-fusion-research') {
        if ([string]::IsNullOrWhiteSpace($ResearchInput)) { throw 'Supply saved -ResearchInput original historical prediction directory; one early-season weight fit only, no downloads or team refits.' }
        & $taskPython -m football_model.market_fusion_research --root $taskRoot --parent $ResearchInput
    }
    elseif ($Action -eq 'laplace-diagnostics') {
        if ([string]::IsNullOrWhiteSpace($ResearchInput)) { throw 'Supply saved -ResearchInput hierarchical XI research directory; arithmetic diagnostics only, no fits.' }
        & $taskPython -m football_model.laplace_prediction_diagnostics --root $taskRoot --research $ResearchInput
    }
    elseif ($Action -eq 'season-state-research') {
        if ([string]::IsNullOrWhiteSpace($InspectionInput)) { throw 'Supply saved -InspectionInput four-season English inspection directory; bounded earlier-season selection, no downloads or current forecast changes.' }
        & $taskPython -m football_model.season_state_research --root $taskRoot --inspection $InspectionInput
    }
    elseif ($Action -eq 'minutes-research') { & $taskPython -m football_model.prematch_minutes_research }
    elseif ($Action -eq 'core-tuning') { & $taskPython -m football_model.core_tuning_experiment }
    elseif ($Action -eq 'score-shape') { & $taskPython -m football_model.score_shape_experiment }
    elseif ($Action -eq 'cmp-research') { & $taskPython -m football_model.cmp_experiment }
    elseif ($Action -eq 'cmp-forward') { & $taskPython -m football_model.cmp_forward_experiment }
    elseif ($Action -eq 'dependence-forward') { & $taskPython -m football_model.bivariate_poisson_experiment }
    elseif ($Action -eq 'rate-calibration-forward') { & $taskPython -m football_model.rate_calibration_forward_experiment }
    elseif ($Action -eq 'snapshot-dynamic') { & $taskPython -m football_model.snapshot_dynamic_experiment }
    elseif ($Action -eq 'prematch-pilot') { & $taskPython -m football_model.prospective_pilot }
    elseif ($Action -eq 'multiline-pilot') { & $taskPython -m football_model.multiline_pilot }
    elseif ($Action -eq 'batch-pilot') {
        if ([string]::IsNullOrWhiteSpace($BatchInput)) {
            $taskBatchPattern = Join-Path $taskRoot 'data\prematch-intelligence\20261024_batch\run_*\batch_case.json'
            $taskBatchSource = Get-ChildItem -Path $taskBatchPattern -File | Sort-Object FullName | Select-Object -Last 1
            if ($null -eq $taskBatchSource) { throw 'No assembled batch case; prepare and record the real forecasts first.' }
            $BatchInput = $taskBatchSource.FullName
        }
        & $taskPython -m football_model.batch_multiline export --root $taskRoot --case $BatchInput
    }
    elseif ($Action -eq 'batch-score') {
        if ([string]::IsNullOrWhiteSpace($BatchInput) -or $ResultEvidence.Count -eq 0) {
            throw 'Supply -BatchInput <saved case> and -ResultEvidence <saved official observations>; no forecasts or results are invented.'
        }
        $taskScoreArguments = @('-m', 'football_model.batch_scoring', '--root', $taskRoot, '--case', $BatchInput)
        foreach ($taskEvidence in $ResultEvidence) { $taskScoreArguments += @('--evidence', $taskEvidence) }
        & $taskPython @taskScoreArguments
    }
    elseif ($Action -eq 'english-tiers-research') {
        if ([string]::IsNullOrWhiteSpace($TierInput)) { throw 'Supply -TierInput <saved English-tier inspection directory>; this research action does not download sources.' }
        & $taskPython -m football_model.english_tiers_experiment --root $taskRoot --inspection $TierInput
    }
    elseif ($Action -eq 'english-tiers-weighting') {
        if ([string]::IsNullOrWhiteSpace($TierInput) -or [string]::IsNullOrWhiteSpace($TierReference)) {
            throw 'Supply saved -TierInput inspection directory and -TierReference result JSON; no implicit downloads or parameter search.'
        }
        & $taskPython -m football_model.english_tiers_weighting --root $taskRoot --inspection $TierInput --reference $TierReference
    }
    elseif ($Action -eq 'player-context') {
        if ([string]::IsNullOrWhiteSpace($BatchInput) -or [string]::IsNullOrWhiteSpace($NewsInput) -or [string]::IsNullOrWhiteSpace($SnapshotInput)) {
            throw 'Supply saved -BatchInput case, -NewsInput news and -SnapshotInput official observation directory; this only prepares descriptive packets.'
        }
        & $taskPython -m football_model.fpl_player_context --root $taskRoot --case $BatchInput --news $NewsInput --snapshot $SnapshotInput
    }
    elseif ($Action -eq 'prospective-revision') {
        if ([string]::IsNullOrWhiteSpace($BatchInput) -or [string]::IsNullOrWhiteSpace($RevisionInput)) {
            throw 'Supply saved -BatchInput original case and -RevisionInput packet directory with real saved LLM responses; no implicit LLM calls or refitting.'
        }
        & $taskPython -m football_model.prospective_revision --root $taskRoot --case $BatchInput --packet-dir $RevisionInput
    }
    elseif ($Action -eq 'impect-pilot') {
        if ([string]::IsNullOrWhiteSpace($SampleInput)) {
            throw 'Supply -SampleInput <saved IMPECT pilot directory>; no implicit downloads or training.'
        }
        & $taskPython -m football_model.impect_pilot --root $taskRoot --sample $SampleInput
    }
    elseif ($Action -eq 'impect-season') {
        if ([string]::IsNullOrWhiteSpace($SeasonInput)) {
            throw 'Supply -SeasonInput <saved IMPECT season directory>; no implicit downloads or training.'
        }
        & $taskPython -m football_model.impect_season --root $taskRoot --source $SeasonInput
    }
    elseif ($Action -eq 'impect-player-research') {
        if ([string]::IsNullOrWhiteSpace($ParsedInput) -or [string]::IsNullOrWhiteSpace($InspectionInput)) {
            throw 'Supply saved -ParsedInput and -InspectionInput JSON; no implicit downloads, repairs or parameter search.'
        }
        & $taskPython -m football_model.impect_player_research --root $taskRoot --parsed $ParsedInput --inspection $InspectionInput
    }
    elseif ($Action -eq 'starting-xi-data') {
        if ([string]::IsNullOrWhiteSpace($SeasonInput)) { throw 'Supply -SeasonInput <saved IMPECT season directory>; no downloads or training.' }
        & $taskPython -m football_model.impect_starting_xi --root $taskRoot --source $SeasonInput
    }
    elseif ($Action -eq 'starting-xi-research') {
        if ([string]::IsNullOrWhiteSpace($ParsedInput) -or [string]::IsNullOrWhiteSpace($InspectionInput)) { throw 'Supply saved -ParsedInput XI matches and -InspectionInput JSON; no implicit data refresh or tuning.' }
        & $taskPython -m football_model.starting_xi_research --root $taskRoot --matches $ParsedInput --inspection $InspectionInput
    }
    elseif ($Action -eq 'starting-xi-diagnostics') {
        if ([string]::IsNullOrWhiteSpace($ParsedInput) -or [string]::IsNullOrWhiteSpace($ResearchInput)) { throw 'Supply saved -ParsedInput XI matches and -ResearchInput prior research directory; no fitting.' }
        & $taskPython -m football_model.starting_xi_diagnostics --root $taskRoot --source $ParsedInput --research $ResearchInput
    }
    elseif ($Action -eq 'starting-roles-data') {
        if ([string]::IsNullOrWhiteSpace($ParsedInput) -or [string]::IsNullOrWhiteSpace($InspectionInput)) { throw 'Supply saved -ParsedInput original XI matches and -InspectionInput original XI inspection JSON; no downloads or training.' }
        & $taskPython -m football_model.impect_starting_roles --root $taskRoot --source-inspection $InspectionInput --matches-source $ParsedInput
    }
    elseif ($Action -eq 'statsbomb-starting-roles') {
        if ([string]::IsNullOrWhiteSpace($SeasonInput) -or [string]::IsNullOrWhiteSpace($InspectionInput) -or [string]::IsNullOrWhiteSpace($ParsedInput)) { throw 'Supply saved -SeasonInput StatsBomb directory, -InspectionInput full catalogue JSON and -ParsedInput independent E0 score CSV; no downloads or fitting.' }
        & $taskPython -m football_model.statsbomb_starting_roles --root $taskRoot --source $SeasonInput --catalogue $InspectionInput --score-file $ParsedInput
    }
    elseif ($Action -eq 'starting-xi-transfer') {
        if ([string]::IsNullOrWhiteSpace($ParsedInput) -or [string]::IsNullOrWhiteSpace($InspectionInput)) { throw 'Supply saved -ParsedInput StatsBomb role matches and -InspectionInput source inspection JSON; one fixed structure, no search or downloads.' }
        & $taskPython -m football_model.starting_xi_transfer --root $taskRoot --source $ParsedInput --inspection $InspectionInput
    }
    elseif ($Action -eq 'hierarchical-xi-research') {
        if ([string]::IsNullOrWhiteSpace($ParsedInput) -or [string]::IsNullOrWhiteSpace($InspectionInput) -or [string]::IsNullOrWhiteSpace($ResearchInput)) { throw 'Supply saved -ParsedInput role-aware matches, -InspectionInput role inspection JSON and -ResearchInput prior XI research directory; fixed scales only, no downloads.' }
        & $taskPython -m football_model.hierarchical_xi_research --root $taskRoot --source $ParsedInput --inspection $InspectionInput --old-research $ResearchInput
    }
    elseif ($Action -eq 'backtest') { & $taskPython -m football_model.real_backtest }
    elseif ($Action -eq 'native-train') { & $taskPython -m football_model.native_experiment }
    elseif ($Action -eq 'forecast-diagnostics') { & $taskPython -m football_model.forecast_diagnostics }
    elseif ($Action -eq 'calibration-research') { & $taskPython -m football_model.calibration_experiment }
    elseif ($Action -eq 'refresh-research') { & $taskPython -m football_model.refresh_experiment }
    elseif ($Action -eq 'goal-rate-diagnostics') { & $taskPython -m football_model.goal_rate_experiment }
    elseif ($Action -eq 'rate-calibration-research') { & $taskPython -m football_model.rate_calibration_experiment }
    elseif ($Action -eq 'schedule-research') { & $taskPython -m football_model.schedule_experiment }
    elseif ($Action -eq 'player-data') { & $taskPython -m football_model.player_probe }
    elseif ($Action -eq 'player-train') { & $taskPython -m football_model.full_season_experiment }
    elseif ($Action -eq 'player-rolling') { & $taskPython -m football_model.rolling_player_experiment }
    elseif ($Action -eq 'catalog-data') { & $taskPython -m football_model.statsbomb_catalog }
    elseif ($Action -eq 'wyscout-data') { & $taskPython -m football_model.wyscout_data }
    elseif ($Action -eq 'wyscout-inspect') { & $taskPython -m football_model.wyscout_probe }
    elseif ($Action -eq 'timing-measure') { & $taskPython -m football_model.timing_precision_experiment }
    elseif ($Action -eq 'timing-fit') { & $taskPython -m football_model.timing_precision_experiment --fit-models }
    elseif ($Action -eq 'second-team') { & $taskPython -m football_model.second_season_team_experiment }
    elseif ($Action -eq 'second-players') { & $taskPython -m football_model.second_season_player_experiment }
    elseif ($Action -eq 'window-clock') { & $taskPython -m football_model.window_clock_experiment }
    elseif ($Action -eq 'player-shrinkage') { & $taskPython -m football_model.regularization_experiment }
    elseif ($Action -eq 'player-forward-selection') { & $taskPython -m football_model.forward_selection_experiment }
    elseif ($Action -eq 'player-exposure-scaling') { & $taskPython -m football_model.scaling_experiment }
    else { & $taskPython -m pytest -q }
    if ($LASTEXITCODE -ne 0) { throw "Model operation failed: $Action" }
} finally { Pop-Location }

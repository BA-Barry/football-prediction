param([ValidateSet('benchmark','tune','forward','all')][string]$Stage='all')
$ErrorActionPreference='Stop'
$taskRoot = if ([string]::IsNullOrWhiteSpace($env:FOOTBALL_MODEL_ROOT)) {
    Split-Path -Parent $PSScriptRoot
} else {
    [System.IO.Path]::GetFullPath($env:FOOTBALL_MODEL_ROOT)
}
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH=Join-Path $taskRoot 'model_v2\src'
$env:TEMP=Join-Path $taskRoot 'work\tmp'
$env:TMP=$env:TEMP
$env:TMPDIR=$env:TEMP
$env:PIP_CACHE_DIR=Join-Path $taskRoot 'work\pip-cache'
$env:NUMBA_CACHE_DIR=Join-Path $taskRoot 'work\numba-cache'
$env:JOBLIB_TEMP_FOLDER=$env:TEMP
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& (Join-Path $taskRoot '.venv\Scripts\python.exe') -B (Join-Path $taskRoot 'work\run_kickscore_20261025.py') --stage $Stage
if ($LASTEXITCODE -ne 0) { throw "KickScore study stopped with exit code $LASTEXITCODE" }

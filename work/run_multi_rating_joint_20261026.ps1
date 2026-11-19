param(
    [switch]$PrepareOnly,
    [switch]$ReusePrepared
)
$ErrorActionPreference = 'Stop'
$projectRoot = if ([string]::IsNullOrWhiteSpace($env:FOOTBALL_MODEL_ROOT)) {
    Split-Path -Parent $PSScriptRoot
} else {
    [System.IO.Path]::GetFullPath($env:FOOTBALL_MODEL_ROOT)
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = "$projectRoot\model_v2\src;$projectRoot\work"
$env:TEMP = "$projectRoot\work\tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$projectRoot\work\pip-cache"
$env:PYTHONUSERBASE = "$projectRoot\work\python-user"
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$runArgs = @('-B', "$projectRoot\work\run_multi_rating_joint_20261026.py")
if ($PrepareOnly) { $runArgs += '--prepare-only' }
if ($ReusePrepared) { $runArgs += '--reuse-prepared' }
Push-Location -LiteralPath $projectRoot
try {
    & "$projectRoot\.venv\Scripts\python.exe" @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Multi-rating run failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }

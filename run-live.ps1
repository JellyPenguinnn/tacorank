<#
.SYNOPSIS
    Build a TacoRank deployment, preflight it, and run the autonomous loop.

.DESCRIPTION
    The Windows counterpart to run-new-live.sh, which resolves the CLI at
    .venv/bin and therefore cannot run here: this repository's virtualenv uses
    the Windows layout (.venv\Scripts\tacorank.exe).

    Reads DEEPSEEK_API_KEY from an .env file if it is not already set in the
    session, and never prints the value.

.PARAMETER RunId
    Run identifier. Defaults to a UTC timestamp.

.PARAMETER MaxExperiments
    Experiment ceiling for this run. Written into the generated run-config.json
    after setup, which is safe because the config hash is computed when the run
    starts, not when the deployment is built.

.PARAMETER Families
    Research families this run may propose. Restricting the set concentrates
    the search instead of spreading it thin across every family.

.PARAMETER EnvFile
    Path to a file containing DEEPSEEK_API_KEY=...

.PARAMETER SkipRun
    Build and preflight only. Useful for checking a deployment before spending
    provider tokens.

.EXAMPLE
    .\run-live.ps1
.EXAMPLE
    .\run-live.ps1 -RunId my_run -MaxExperiments 15
.EXAMPLE
    .\run-live.ps1 -Families objective,temporal_history,model
#>
[CmdletBinding()]
param(
    [string]$RunId = ("run_{0}" -f (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")),
    [int]$MaxExperiments = 15,
    [string[]]$Families = @("model", "objective", "temporal_history", "duration_bias", "ensemble"),
    [string]$EnvFile = "..\.env",
    # Where per-run Trae runtimes are created. The coding worker refuses to
    # start if a .env is reachable from the runtime root upwards, so keeping
    # runtimes outside the directory holding your credentials satisfies that
    # check by layout rather than by moving the file.
    [string]$RuntimeRoot = "",
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Fail([string]$Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

function Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

$repo = $PSScriptRoot
Set-Location $repo

# --- CLI -------------------------------------------------------------------
$tacorank = Join-Path $repo ".venv\Scripts\tacorank.exe"
if (-not (Test-Path $tacorank)) {
    Fail "could not find $tacorank; create the virtualenv and pip install -e . first"
}

# --- Credential ------------------------------------------------------------
# Only read the file when the session does not already carry the key, and
# never echo the value.
if (-not $env:DEEPSEEK_API_KEY) {
    $resolvedEnv = if ([System.IO.Path]::IsPathRooted($EnvFile)) { $EnvFile } else { Join-Path $repo $EnvFile }
    if (-not (Test-Path $resolvedEnv)) {
        Fail "DEEPSEEK_API_KEY is not set and no env file at $resolvedEnv (pass -EnvFile)"
    }
    foreach ($line in Get-Content $resolvedEnv) {
        if ($line -match '^\s*(?<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?<v>.*)$') {
            $value = $Matches['v'].Trim().Trim('"').Trim("'")
            if ($Matches['k'] -eq 'DEEPSEEK_API_KEY' -and $value) {
                $env:DEEPSEEK_API_KEY = $value
            }
        }
    }
    if (-not $env:DEEPSEEK_API_KEY) { Fail "no DEEPSEEK_API_KEY entry found in $resolvedEnv" }
    Write-Host "loaded DEEPSEEK_API_KEY from $resolvedEnv (length $($env:DEEPSEEK_API_KEY.Length))"
}

# --- Toolchain -------------------------------------------------------------
$python312 = $env:TACORANK_PYTHON312
if (-not $python312) {
    $python312 = (Get-Command py -ErrorAction SilentlyContinue) `
        | ForEach-Object { & $_.Source -3.12 -c "import sys; print(sys.executable)" 2>$null } `
        | Select-Object -First 1
}
if (-not $python312 -or -not (Test-Path $python312)) {
    Fail "could not locate Python 3.12; set TACORANK_PYTHON312 to its full path"
}

$docker = (Get-Command docker -ErrorAction SilentlyContinue)
if (-not $docker) { Fail "docker was not found on PATH; start Docker Desktop" }
$dockerPath = $docker.Source

# --- Cleanliness -----------------------------------------------------------
# setup-live refuses a dirty tree, and counts ignored files inside the pinned
# submodule as dirty. Python imports leave bytecode that ordinary git status
# hides, so clear exactly that and nothing else.
Step "Checking repository cleanliness"
$tracked = git status --porcelain=v1 | Where-Object { $_ -notmatch '^\?\?' }
if ($tracked) {
    $tracked | Write-Host
    Fail "tracked changes are present; commit or stash them before a live run"
}
git -C kuairand-starter-kit clean -q -fdX -- __pycache__ '*.pyc' '*.pyo' | Out-Null
$submodule = git -C kuairand-starter-kit status --porcelain=v1 --untracked-files=all --ignored=matching
if ($submodule) {
    $submodule | Write-Host
    Fail "kuairand-starter-kit contains non-bytecode changes; preserving them"
}
Write-Host "clean"

# --- Paths -----------------------------------------------------------------
$deploymentDir = Join-Path $repo ".tacorank\deployments\$RunId"
$runtimeBase   = if ($RuntimeRoot) { $RuntimeRoot } else { Join-Path (Split-Path $repo -Parent) ".tacorank-runtime" }
$runtimeDir    = Join-Path $runtimeBase "tacorank-$RunId"
$runDir        = Join-Path $repo "runs\$RunId"
foreach ($path in @($deploymentDir, $runtimeDir, $runDir)) {
    if (Test-Path $path) { Fail "already exists for run id '$RunId': $path (choose another -RunId)" }
}
$config     = Join-Path $deploymentDir "run-config.json"
$liveConfig = Join-Path $deploymentDir "live-adapters.json"

# The coding worker rejects any .env discoverable from the runtime root
# upwards, so that Trae cannot pick up unreviewed credentials. Check it now
# rather than after the Docker build: preflight would otherwise fail with
# TRAE_DOTENV_FORBIDDEN several minutes in.
$ancestor = Split-Path $runtimeDir -Parent
while ($ancestor) {
    $stray = Join-Path $ancestor ".env"
    if (Test-Path $stray) {
        Write-Host "error: '$stray' sits on the Trae runtime search path." -ForegroundColor Red
        Write-Host "The coding worker refuses to start while an unreviewed .env is reachable" -ForegroundColor Red
        Write-Host "from the runtime root, so that it cannot read your credentials." -ForegroundColor Red
        Write-Host "Move or rename that file (a name other than '.env' is enough, since" -ForegroundColor Red
        Write-Host "python-dotenv only discovers that exact name) and pass -EnvFile to it." -ForegroundColor Red
        exit 1
    }
    $parent = Split-Path $ancestor -Parent
    if ($parent -eq $ancestor) { break }
    $ancestor = $parent
}

Write-Host ""
Write-Host "run id          : $RunId"
Write-Host "max experiments : $MaxExperiments"
Write-Host "families        : $($Families -join ', ')"

# --- Build -----------------------------------------------------------------
Step "Building deployment (rebuilds the Docker image; this is the slow step)"
& $tacorank setup-live `
    --repository-root $repo `
    --deployment-dir $deploymentDir `
    --runtime-dir $runtimeDir `
    --data-dir (Join-Path $repo "KuaiRand-Pure\data") `
    --python312 $python312 `
    --docker $dockerPath `
    --run-id $RunId `
    --download-data
if ($LASTEXITCODE -ne 0) { Fail "setup-live failed with exit code $LASTEXITCODE" }
if (-not (Test-Path $config)) { Fail "setup-live did not produce $config" }

# --- Per-run configuration -------------------------------------------------
# Safe to edit here: the config hash is computed at run start, so preflight and
# run both see these values. Editing a config mid-run would break the ledger.
Step "Applying per-run configuration"
$runConfig = Get-Content $config -Raw | ConvertFrom-Json
$runConfig.max_experiments = $MaxExperiments
$runConfig.allowed_research_families = $Families
# Write UTF-8 without a BOM. Set-Content -Encoding utf8 and Out-File both emit
# one under Windows PowerShell 5.1, and the config loader rejects it with
# "Unexpected UTF-8 BOM (decode using utf-8-sig)" before preflight can run.
$json = $runConfig | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($config, $json, (New-Object System.Text.UTF8Encoding $false))
Write-Host "max_experiments = $($runConfig.max_experiments)"
Write-Host "allowed_research_families = $($runConfig.allowed_research_families -join ', ')"

# --- Preflight -------------------------------------------------------------
Step "Preflight (verifies prerequisites; creates no ledger)"
& $tacorank preflight --config $config --live-config $liveConfig
if ($LASTEXITCODE -ne 0) { Fail "preflight failed with exit code $LASTEXITCODE" }

if ($SkipRun) {
    Write-Host ""
    Write-Host "Preflight passed. Skipping the run because -SkipRun was given." -ForegroundColor Green
    Write-Host "To run it later:"
    Write-Host "  .venv\Scripts\tacorank.exe run --config $config --live-config $liveConfig"
    exit 0
}

# --- Run -------------------------------------------------------------------
Step "Running the autonomous loop"
Write-Host "Watch it from a second terminal with:"
Write-Host "  python scripts\watch_run.py --run-id $RunId"
Write-Host ""
& $tacorank run --config $config --live-config $liveConfig
$runExit = $LASTEXITCODE

# --- Result ----------------------------------------------------------------
Step "Final status"
& $tacorank status --run-id $RunId --repository-root $repo
python (Join-Path $repo "scripts\watch_run.py") --run-id $RunId --once

if ($runExit -ne 0) {
    Write-Host ""
    Write-Host "The loop exited with code $runExit." -ForegroundColor Yellow
    Write-Host "If it stopped on a resumable checkpoint, continue it with:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\tacorank.exe resume --config $config --live-config $liveConfig"
    exit $runExit
}

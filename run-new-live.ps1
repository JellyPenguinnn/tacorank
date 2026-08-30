[CmdletBinding()]
param(
    # Use -DownloadData for a fresh clone or when KuaiRand-Pure/data is incomplete.
    [switch]$DownloadData,
    [switch]$SkipInstall,
    [string]$RunId,
    [string]$Python312,
    [string]$Docker
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

function Fail([string]$Message) {
    throw "error: $Message"
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "command failed (exit $LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir ".")).Path
$gitRootText = (& git -C $repoRoot rev-parse --show-toplevel 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitRootText)) {
    Fail "the script must run inside a Git repository"
}
$gitRoot = (Resolve-Path $gitRootText).Path
if ($gitRoot -ne $repoRoot) {
    Fail "run this script from the TacoRank repository checkout"
}

Push-Location $repoRoot
$lockDir = Join-Path $repoRoot ".tacorank\live-run.lock"
$lockOwned = $false
try {
    # setup-live binds the deployment to an exact clean commit. Do this check
    # before creating any generated directories or downloading data.
    $trackedChanges = (& git status --porcelain=v1 --untracked-files=no | Out-String).Trim()
    if (-not [string]::IsNullOrWhiteSpace($trackedChanges)) {
        & git status --short
        Fail "tracked changes are present; commit or preserve them before live setup"
    }

    $starterKit = Join-Path $repoRoot "kuairand-starter-kit"
    if (-not (Test-Path (Join-Path $starterKit ".git"))) {
        Invoke-Checked "git" @("submodule", "update", "--init", "--recursive", "--", "kuairand-starter-kit")
    }
    $expectedStarterCommit = (& git -C $repoRoot rev-parse "HEAD:kuairand-starter-kit" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { Fail "could not resolve the pinned kuairand-starter-kit commit" }
    $actualStarterCommit = (& git -C $starterKit rev-parse HEAD 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) { Fail "kuairand-starter-kit is not a usable Git worktree" }
    if ($actualStarterCommit -ne $expectedStarterCommit) {
        Fail "kuairand-starter-kit is not at the pinned commit"
    }
    $starterChanges = (& git -C $starterKit status --porcelain=v1 --untracked-files=all | Out-String).Trim()
    if (-not [string]::IsNullOrWhiteSpace($starterChanges)) {
        & git -C $starterKit status --short --untracked-files=all
        Fail "kuairand-starter-kit has tracked or untracked changes; preserving them"
    }
    # Imports can leave ignored bytecode that setup-live deliberately treats
    # as dirty. Remove only that reproducible cache, never other ignored files.
    Invoke-Checked "git" @("-C", $starterKit, "clean", "-q", "-fdX", "--", "__pycache__", "*.pyc", "*.pyo")
    $strictStarterStatus = (& git -C $starterKit status --porcelain=v1 --untracked-files=all --ignored=matching | Out-String).Trim()
    if (-not [string]::IsNullOrWhiteSpace($strictStarterStatus)) {
        & git -C $starterKit status --short --untracked-files=all --ignored=matching
        Fail "kuairand-starter-kit contains non-bytecode generated files"
    }

    if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
        Fail "DEEPSEEK_API_KEY must be exported in this PowerShell session (the value is never printed)"
    }

    $controlPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    $taco = Join-Path $repoRoot ".venv\Scripts\tacorank.exe"
    if (-not (Test-Path $controlPython) -or -not (Test-Path $taco)) {
        if ($SkipInstall) {
            Fail "control-plane .venv is incomplete; remove -SkipInstall or install it first"
        }
        if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
            Fail "Python Launcher (py.exe) was not found; install Python 3.12 or pass a prepared .venv"
        }
        Invoke-Checked "py.exe" @("-3.12", "-m", "venv", ".venv")
    }
    if (-not $SkipInstall) {
        Invoke-Checked $controlPython @("-m", "pip", "install", "-r", "requirements-dev.txt")
        Invoke-Checked $controlPython @("-m", "pip", "install", "--no-deps", "-e", ".")
    }
    if (-not (Test-Path $taco)) { Fail "the TacoRank CLI was not installed in .venv" }

    if ([string]::IsNullOrWhiteSpace($Python312)) {
        if (-not (Get-Command py.exe -ErrorAction SilentlyContinue)) {
            Fail "Python 3.12 was not found; pass -Python312 C:\path\to\python.exe"
        }
        $Python312 = (& py.exe -3.12 -c "import sys; print(sys.executable)" | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Python312)) {
            Fail "Python 3.12 was not found; pass -Python312 C:\path\to\python.exe"
        }
    }
    $Python312 = (Resolve-Path $Python312).Path
    Invoke-Checked $Python312 @("--version")

    if ([string]::IsNullOrWhiteSpace($Docker)) {
        $dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
        if (-not $dockerCommand) { $dockerCommand = Get-Command docker -ErrorAction SilentlyContinue }
        if (-not $dockerCommand) { Fail "Docker was not found; start Docker Desktop or pass -Docker C:\path\to\docker.exe" }
        $Docker = $dockerCommand.Source
    }
    $Docker = (Resolve-Path $Docker).Path
    & $Docker info --format '{{.ServerVersion}}' *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "Docker daemon is not reachable; start Docker Desktop in Linux-container mode"
    }

    if ([string]::IsNullOrWhiteSpace($RunId)) {
        $RunId = "run_{0}_{1}" -f [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ"), $PID
    }
    if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$') {
        Fail "RunId must contain only letters, digits, '.', '_' or '-' and be at most 64 characters"
    }
    $deploymentDir = Join-Path $repoRoot ".tacorank\deployments\$RunId"
    $repoParent = (Get-Item $repoRoot).Parent.FullName
    $runtimeDir = Join-Path $repoParent ".tacorank-runtime\$((Get-Item $repoRoot).Name)-$RunId"
    $runDir = Join-Path $repoRoot "runs\$RunId"
    if ((Test-Path $deploymentDir) -or (Test-Path $runtimeDir) -or (Test-Path $runDir)) {
        Fail "run identity already exists; choose a new -RunId (previous evidence is immutable)"
    }

    $lockParent = Split-Path -Parent $lockDir
    New-Item -ItemType Directory -Force -Path $lockParent | Out-Null
    try {
        New-Item -ItemType Directory -Path $lockDir -ErrorAction Stop | Out-Null
    } catch {
        Fail "another script-managed live run is active or the lock exists at $lockDir; inspect it before removal"
    }
    $lockOwned = $true
    Set-Content -LiteralPath (Join-Path $lockDir "pid") -Value $PID -NoNewline

    $dataDir = Join-Path $repoRoot "KuaiRand-Pure\data"
    $config = Join-Path $deploymentDir "run-config.json"
    $liveConfig = Join-Path $deploymentDir "live-adapters.json"
    Write-Output "Starting new TacoRank live run: $RunId"
    Write-Output "Repository: $repoRoot"

    $setupArgs = @(
        "setup-live", "--repository-root", $repoRoot,
        "--deployment-dir", $deploymentDir, "--runtime-dir", $runtimeDir,
        "--data-dir", $dataDir, "--python312", $Python312,
        "--docker", $Docker, "--run-id", $RunId
    )
    if ($DownloadData) { $setupArgs += "--download-data" }
    Invoke-Checked $taco $setupArgs

    # setup imports may have generated ignored bytecode in the submodule.
    Invoke-Checked "git" @("-C", $starterKit, "clean", "-q", "-fdX", "--", "__pycache__", "*.pyc", "*.pyo")
    $strictAfterSetup = (& git -C $starterKit status --porcelain=v1 --untracked-files=all --ignored=matching | Out-String).Trim()
    if (-not [string]::IsNullOrWhiteSpace($strictAfterSetup)) { Fail "setup generated unexpected submodule files" }

    Invoke-Checked $taco @("preflight", "--config", $config, "--live-config", $liveConfig)
    if (Test-Path (Join-Path $runDir "events.jsonl")) { Fail "preflight unexpectedly created a ledger" }
    Invoke-Checked $taco @("run", "--config", $config, "--live-config", $liveConfig)

    $statusLines = & $taco "status" "--run-id" $RunId "--repository-root" $repoRoot
    if ($LASTEXITCODE -ne 0) { Fail "could not read final run status" }
    $status = ($statusLines -join "`n") | ConvertFrom-Json
    Write-Output ($statusLines -join "`n")
    if ($status.status -ne "finalized" -or $status.phase -ne "finalized") {
        Fail "run did not finalize: status=$($status.status) phase=$($status.phase)"
    }
    if ([string]::IsNullOrWhiteSpace([string]$status.final_experiment_id)) {
        Fail "run finalized without final_experiment_id"
    }
    Invoke-Checked $taco @("validate-ledger", "--run-id", $RunId, "--repository-root", $repoRoot)
    Invoke-Checked $taco @("rebuild-views", "--run-id", $RunId, "--repository-root", $repoRoot)
    Write-Output "TacoRank live run completed and validated: $RunId"
}
finally {
    if ($lockOwned -and (Test-Path $lockDir)) {
        Remove-Item -LiteralPath $lockDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}

[CmdletBinding()]
param(
    [ValidateRange(1, 10)]
    [int]$Trials = 3,

    [string]$ClawRoot = "D:\claw-eval",

    [switch]$Smoke,

    [ValidateSet("DM001", "DM002", "DM003", "DM004", "DM005", "DM006")]
    [string]$TaskId,

    [switch]$BatchTask,

    [switch]$Mock,

    [switch]$NoJudge,

    [switch]$Docker,

    [string]$DockerApiUrl = "http://127.0.0.1:9310/api/v1"
)

$ErrorActionPreference = "Stop"
$DataMindRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TasksRoot = Join-Path $DataMindRoot "benchmarks\claw_eval\tasks"
$ConfigPath = Join-Path $PSScriptRoot "config.datamind.yaml"
$ClawPythonEnv = Join-Path $ClawRoot ".venv"
$ExpectedTasks = @("DM001", "DM002", "DM003", "DM004", "DM005", "DM006")

if ($Smoke -and ($TaskId -or $BatchTask)) {
    throw "-Smoke cannot be combined with -TaskId or -BatchTask."
}
if ($BatchTask -and -not $TaskId) {
    throw "-BatchTask requires -TaskId."
}

if (-not (Test-Path -LiteralPath $ClawRoot)) {
    throw "claw-eval directory does not exist: $ClawRoot"
}
if (-not (Test-Path -LiteralPath $ClawPythonEnv)) {
    throw "claw-eval environment does not exist: $ClawPythonEnv"
}

$ActualTasks = @(
    Get-ChildItem -LiteralPath $TasksRoot -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "task.yaml") } |
        Select-Object -ExpandProperty Name |
        Sort-Object
)
if (Compare-Object -ReferenceObject $ExpectedTasks -DifferenceObject $ActualTasks) {
    throw "Unexpected task set in ${TasksRoot}: $($ActualTasks -join ', ')"
}

$EvaluationPorts = if ($Docker) { @(9320) } else { @(9310, 9320) }
$Listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in $EvaluationPorts }
if ($Listeners) {
    $Details = ($Listeners | ForEach-Object { "$($_.LocalPort)/pid=$($_.OwningProcess)" }) -join ", "
    throw "DataMind evaluation ports are already occupied: $Details"
}

if (-not $Docker -and -not $Mock -and -not (Test-Path Env:DATAMIND_KIMI_API_KEY)) {
    throw "DATAMIND_KIMI_API_KEY is not set in this PowerShell session."
}
if (-not $Mock -and -not $NoJudge -and -not (Test-Path Env:CLAW_EVAL_JUDGE_API_KEY)) {
    throw "CLAW_EVAL_JUDGE_API_KEY is not set in this PowerShell session."
}

$PreviousJudgeBaseUrl = $env:CLAW_EVAL_JUDGE_BASE_URL
$PreviousJudgeModelId = $env:CLAW_EVAL_JUDGE_MODEL_ID
if (-not $NoJudge) {
    # Match config.datamind.yaml and expose the selected judge to trace
    # summarizers. The API key remains environment-only.
    $env:CLAW_EVAL_JUDGE_BASE_URL = "https://api.moonshot.cn/v1"
    $env:CLAW_EVAL_JUDGE_MODEL_ID = "kimi-k3"
}

$PreviousProvider = $env:DATAMIND_EVAL_LLM_PROVIDER
$PreviousApiUrl = $env:DATAMIND_EVAL_API_URL
$PreviousAuthMode = $env:DATAMIND_EVAL_AUTH_MODE
$env:DATAMIND_KIMI_BASE_URL = "https://api.moonshot.cn/v1"
$env:DATAMIND_KIMI_MODEL = "kimi-k2.6"
$env:DATAMIND_EVAL_LLM_PROVIDER = if ($Mock) { "mock" } else { "kimi" }
if ($Docker) {
    $env:DATAMIND_EVAL_API_URL = $DockerApiUrl
    $env:DATAMIND_EVAL_AUTH_MODE = "session"
}
if ($Mock) {
    $NoJudge = $true
}

$Arguments = @(
    "run",
    "--no-capture-output",
    "--prefix", $ClawPythonEnv,
    "claw-eval"
)
if ($Smoke -or ($TaskId -and -not $BatchTask)) {
    $SelectedTask = if ($TaskId) { $TaskId } else { "DM001" }
    $SelectedTrials = if ($Smoke) { 1 } else { $Trials }
    $Arguments += @(
        "run",
        "--task", (Join-Path $TasksRoot $SelectedTask),
        "--config", $ConfigPath,
        "--trials", "$SelectedTrials"
    )
} else {
    $Arguments += @(
        "batch",
        "--tasks-dir", $TasksRoot,
        "--config", $ConfigPath,
        "--trials", "$Trials",
        "--parallel", "1"
    )
    if ($TaskId) {
        $Arguments += @("--filter", $TaskId)
    }
}
if ($NoJudge) {
    $Arguments += "--no-judge"
}

Write-Host "DataMind tasks: $(if ($TaskId) { $TaskId } elseif ($Smoke) { 'DM001 (smoke)' } else { $ExpectedTasks -join ', ' })"
Write-Host "Provider: $env:DATAMIND_EVAL_LLM_PROVIDER"
Write-Host "Runtime: $(if ($Docker) { "Docker ($DockerApiUrl)" } else { 'local' })"
Write-Host "Judge: $(if ($NoJudge) { 'disabled' } else { $env:CLAW_EVAL_JUDGE_MODEL_ID })"
Write-Host "Transient retries: $(if ($env:DATAMIND_LLM_TRANSIENT_RETRIES) { $env:DATAMIND_LLM_TRANSIENT_RETRIES } else { '4' }) (backoff base $(if ($env:DATAMIND_LLM_RETRY_BACKOFF_SECONDS) { $env:DATAMIND_LLM_RETRY_BACKOFF_SECONDS } else { '2' })s)"
Write-Host "Trials: $(if ($Smoke) { 1 } else { $Trials })"

Push-Location $ClawRoot
try {
    & conda @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "claw-eval exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
    if ($null -eq $PreviousProvider) {
        Remove-Item Env:DATAMIND_EVAL_LLM_PROVIDER -ErrorAction SilentlyContinue
    } else {
        $env:DATAMIND_EVAL_LLM_PROVIDER = $PreviousProvider
    }
    if ($null -eq $PreviousApiUrl) {
        Remove-Item Env:DATAMIND_EVAL_API_URL -ErrorAction SilentlyContinue
    } else {
        $env:DATAMIND_EVAL_API_URL = $PreviousApiUrl
    }
    if ($null -eq $PreviousAuthMode) {
        Remove-Item Env:DATAMIND_EVAL_AUTH_MODE -ErrorAction SilentlyContinue
    } else {
        $env:DATAMIND_EVAL_AUTH_MODE = $PreviousAuthMode
    }
    if ($null -eq $PreviousJudgeBaseUrl) {
        Remove-Item Env:CLAW_EVAL_JUDGE_BASE_URL -ErrorAction SilentlyContinue
    } else {
        $env:CLAW_EVAL_JUDGE_BASE_URL = $PreviousJudgeBaseUrl
    }
    if ($null -eq $PreviousJudgeModelId) {
        Remove-Item Env:CLAW_EVAL_JUDGE_MODEL_ID -ErrorAction SilentlyContinue
    } else {
        $env:CLAW_EVAL_JUDGE_MODEL_ID = $PreviousJudgeModelId
    }
}

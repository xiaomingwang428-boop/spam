param(
    [int]$Limit = 1000,
    [int]$Timeout = 300,
    [int]$Attempts = 3,
    [int]$ShotsPerClass = 2,
    [int]$MaxConsecutiveFailures = 5,
    [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not $env:ARK_API_KEY) {
    throw "ARK_API_KEY is not set. Set it in this PowerShell session before running."
}

if (-not $env:ARK_BASE_URL) {
    $env:ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
}

if (-not $env:DOUBAO_MODEL) {
    $env:DOUBAO_MODEL = "doubao-seed-2-1-pro-260628"
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$arguments = @(
    "llm_chat_email_classifier.py",
    "--provider", "doubao",
    "--limit", "$Limit",
    "--few-shot",
    "--shots-per-class", "$ShotsPerClass",
    "--timeout", "$Timeout",
    "--attempts", "$Attempts",
    "--max-consecutive-failures", "$MaxConsecutiveFailures"
)

if (-not $Strict) {
    $arguments += "--skip-failed"
}

Write-Host "Running Doubao few-shot sample experiment..."
Write-Host "Limit=$Limit Timeout=$Timeout Attempts=$Attempts ShotsPerClass=$ShotsPerClass SkipFailed=$(-not $Strict) MaxConsecutiveFailures=$MaxConsecutiveFailures"
& $python @arguments

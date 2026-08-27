param(
    [int]$Limit = 1000,
    [int]$Timeout = 300,
    [int]$Attempts = 6,
    [int]$MaxChars = 6000,
    [int]$MaxOutputTokens = 600,
    [switch]$FewShot,
    [int]$ShotsPerClass = 2,
    [switch]$Compact,
    [switch]$SkipFailed
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not $env:OPENAI_API_KEY) {
    throw "OPENAI_API_KEY is not set. Set it in this PowerShell session before running."
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$arguments = @(
    "llm_chat_email_classifier.py",
    "--provider", "openai",
    "--limit", "$Limit",
    "--batch-size", "1",
    "--max-chars", "$MaxChars",
    "--timeout", "$Timeout",
    "--attempts", "$Attempts",
    "--max-output-tokens", "$MaxOutputTokens"
)

if ($Compact) {
    $arguments += "--compact"
}

if ($FewShot) {
    $arguments += "--few-shot"
    $arguments += "--shots-per-class"
    $arguments += "$ShotsPerClass"
}

if ($SkipFailed) {
    $arguments += "--skip-failed"
}

Write-Host "Running GPT small-sample experiment..."
Write-Host "Limit=$Limit Timeout=$Timeout Attempts=$Attempts MaxChars=$MaxChars Compact=$Compact FewShot=$FewShot ShotsPerClass=$ShotsPerClass"
& $python @arguments

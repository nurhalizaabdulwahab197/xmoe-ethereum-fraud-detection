param(
    [ValidateSet('preflight','smoke','full','summarize')]
    [string]$Stage = 'preflight',
    [string]$Graph = '',
    [int]$Threads = 24,
    [switch]$IncludeGpu
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $env:USERPROFILE 'Documents\xmoe-cpu-reproduction\.venv-cpu\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) {
    throw "Existing CPU Python environment not found: $pythonPath. Use the original reproduction PC/environment."
}
$arguments = @((Join-Path $projectRoot 'experiments\rq1_parity\run.py'), '--stage', $Stage,
    '--output', (Join-Path $projectRoot 'results\new_runs\rq1_journal_parity'), '--threads', "$Threads")
if ($Graph) { $arguments += @('--graph', $Graph) }
if ($IncludeGpu) { $arguments += '--include-gpu' }
Write-Host "Running RQ1 stage: $Stage. Original CPU environment; separate experiment outputs."
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw "RQ1 $Stage failed. Preserve the error/log; do not start the next stage." }

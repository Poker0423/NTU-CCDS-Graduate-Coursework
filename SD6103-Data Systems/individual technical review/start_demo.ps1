[CmdletBinding()]
param([switch]$CheckOnly,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
& './.venv/Scripts/python.exe' './scripts/preflight.py' --check-reset
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. Close DuckDB; run reset_demo.ps1; retry.' }
if ($CheckOnly) { return }
if (-not $NoBrowser) {
    $page = ([Uri](Join-Path $PSScriptRoot 'presentation/index.html')).AbsoluteUri
    $edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
    if (Test-Path -LiteralPath $edge) { Start-Process -FilePath $edge -ArgumentList @('--new-window', $page) | Out-Null }
    else { Start-Process $page | Out-Null }
}
try { $Host.UI.RawUI.WindowTitle='SD6103 | DuckDB live SQL' } catch { }
if (-not [Console]::IsOutputRedirected) { try { Clear-Host } catch { } }
Write-Host 'SD6103 | Retail Sales Analytics | Synthetic data | seed 6103' -ForegroundColor Cyan
Write-Host 'Five live commands: .read sql/01_intro.sql through .read sql/05_analyze.sql'
Write-Host 'Exit using .quit before reset/verify.'
& './tools/duckdb.exe' -no-agent -no-init -init 'sql/session.cli' 'database/review.duckdb'
if ($LASTEXITCODE -ne 0) { throw 'DuckDB exited with an error; see troubleshooting guide.' }

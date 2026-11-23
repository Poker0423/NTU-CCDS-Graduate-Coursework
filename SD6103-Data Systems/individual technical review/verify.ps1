[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
& './.venv/Scripts/python.exe' './scripts/validate_demo.py'
if ($LASTEXITCODE -ne 0) { throw 'Verification FAILED; see verification/test_report.md' }
Write-Host 'VERIFICATION PASS. Close any DuckDB session, reset, then start.' -ForegroundColor Green

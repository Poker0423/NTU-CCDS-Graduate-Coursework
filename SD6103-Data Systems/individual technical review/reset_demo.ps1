[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
& './.venv/Scripts/python.exe' './scripts/reset_demo.py'
if ($LASTEXITCODE -ne 0) { throw 'Reset failed. Close other DuckDB sessions, then retry. No force unlock attempted.' }

[CmdletBinding()]
param([string]$PythonExe)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    $candidates = @($PythonExe)
    foreach ($commandName in @('python','python3')) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($command) { $candidates += $command.Source }
    }
    $basePython = $null
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            & $candidate -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $basePython=$candidate; break }
        }
    }
    if (-not $basePython) { throw 'Need Python >=3.11. Supply -PythonExe C:/path/to/python.exe. No global Python modified.' }
    & $basePython -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
@('tools/downloads','tools/wheels','verification','database','data/raw','data/parquet') | ForEach-Object {
    New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot $_) -Force | Out-Null
}
$archive = Join-Path $PSScriptRoot 'tools/downloads/duckdb-cli-windows-amd64.tar.gz'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest 'https://install.duckdb.org/v2.0.0/duckdb-cli-windows-amd64.tar.gz' -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne '91663c2f36c2a334b0e5e639006d8c29c29e013af0c477d80bb931df68870312') {
    throw 'Official archive checksum mismatch'
}
if (-not (Test-Path -LiteralPath './tools/duckdb.exe')) {
    tar -xzf $archive -C './tools'
    if ($LASTEXITCODE -ne 0) { throw 'CLI extraction failed' }
}
$wheel = Get-ChildItem -LiteralPath './tools/wheels' -Filter 'duckdb-2.0.0-*-win_amd64.whl'
if (-not $wheel) {
    & $projectPython -m pip download --no-cache-dir --only-binary=:all: --dest './tools/wheels' -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'DuckDB wheel download failed' }
}
& $projectPython -m pip install --no-cache-dir --no-index --find-links './tools/wheels' -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Project-only dependency install failed' }
& $projectPython -c 'import duckdb; assert tuple(map(int, duckdb.__version__.split(chr(46)))) == (2,0,0)'
if ($LASTEXITCODE -ne 0) { throw 'Python DuckDB version mismatch' }
& './tools/duckdb.exe' -no-agent -no-init --version
if ($LASTEXITCODE -ne 0) { throw 'CLI could not run. See docs/TROUBLESHOOTING.md' }
& './scripts/capture_environment.ps1'
& $projectPython './scripts/generate_data.py'
if ($LASTEXITCODE -ne 0) { throw 'Data generation failed' }
& $projectPython './scripts/build_database.py'
if ($LASTEXITCODE -ne 0) { throw 'Database build failed. Close DuckDB before setup.' }
& './verify.ps1'
& './reset_demo.ps1'
Write-Host 'SETUP PASS. Next: ./start_demo.ps1' -ForegroundColor Green

# Records the local toolchain versions to verification/environment_versions.txt (git-ignored).
$ErrorActionPreference='Stop'
$root = Split-Path -Parent $PSScriptRoot
$report = Join-Path $root 'verification/environment_versions.txt'
$lines = @("Checked (UTC): $([DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm:ss'))", "PowerShell: $($PSVersionTable.PSVersion)", "Process architecture: $([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture)")
$lines += Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture | Format-List | Out-String
$lines += & "$root/.venv/Scripts/python.exe" "$root/scripts/print_versions.py"
$lines += & "$root/.venv/Scripts/python.exe" -m pip --version
$lines += & "$root/tools/duckdb.exe" -no-agent -no-init --version
$lines | Set-Content -LiteralPath $report -Encoding UTF8

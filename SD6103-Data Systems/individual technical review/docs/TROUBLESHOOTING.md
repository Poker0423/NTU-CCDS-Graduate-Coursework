# Troubleshooting

## Paths and entry points

Run every PowerShell entry point from the repository root; each script changes to its own directory, and SQL file paths are relative to the root. The CLI is the local `tools/duckdb.exe` downloaded by `setup.ps1`; a global `duckdb` command is not required. Python is the project `.venv`.

## Script execution policy

If the current shell refuses to run a script, start it in a fresh process:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_demo.ps1
```

This does not change the machine's persistent execution policy. No administrator rights are required.

## Lock conflicts

DuckDB allows one read-write process per database file. Exit the demo CLI with `.quit` before running `setup.ps1`, `verify.ps1` or `reset_demo.ps1`. The scripts fail on lock conflicts; they never force-unlock, delete WAL files or terminate other processes.

## Missing temporary view / export

Run 01, 02, 03, 04, 05 in order in the same CLI session. `retail` is a temporary view created by 02, and 03 creates `data/parquet/retail.parquet`. `reset_demo.ps1` removes only that demo output; the persistent tables, the `retail_base` view and `retail_reference.parquet` are kept.

## Dependencies

DuckDB 2.0 needs Python >= 3.11 (tested with 3.12). If `setup.ps1` cannot find a suitable interpreter, pass one explicitly:

```powershell
.\setup.ps1 -PythonExe C:\path\to\python.exe
```

The official Windows CLI needs the Microsoft VC++ x64 runtime; see the DuckDB installation documentation if it is missing. No Docker, WSL, PostgreSQL or other database service is needed.

## Python SQL fallback

If the official CLI cannot start, a minimal interactive Python shell accepts the same commands and runs the same SQL against the same database:

```powershell
.\.venv\Scripts\python.exe .\scripts\python_sql_shell.py
```

## Plans and measurements

DuckDB 2.0 uses human-readable plan labels; `READ_PARQUET` identifies the Parquet scanner. Small per-operator times can round to zero. Interpret scan output rows, filters, projections and the row-group count; estimates in `EXPLAIN` can differ from the actual counts in `EXPLAIN ANALYZE`. If an assertion fails, inspect `verification/test_report.md` and the corresponding plan file.

## Presentation

Open `presentation/index.html` in Edge or Chrome. Keys: 1-5 or arrows (slides), F (fullscreen), H (hide hints), C (command panel). The page loads no remote resources and runs no SQL.

## Memory

The CLI session is limited to two threads and 768 MB managed memory (`sql/session.cli`). The dataset is small; any recent laptop is sufficient.

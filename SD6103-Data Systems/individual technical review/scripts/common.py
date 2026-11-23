from pathlib import Path
import os
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'database/review.duckdb'

def enter():
    os.chdir(ROOT)

def connect(path=DB, read_only=False):
    enter()
    c = duckdb.connect(str(path), read_only=read_only)
    c.execute("SET threads=2")
    c.execute("SET memory_limit='768MB'")
    c.execute("SET enable_progress_bar=false")
    c.execute("SET autoinstall_known_extensions=false")
    c.execute("SET autoload_known_extensions=false")
    return c

def run_file(c, name):
    """Execute SQL only; CLI presentation commands begin at column zero with '.'"""
    sql = '\n'.join(x for x in (ROOT/name).read_text(encoding='utf-8').splitlines()
                    if not x.startswith('.'))
    results = []
    for statement in c.extract_statements(sql):
        c.execute(statement)
        results.append((c.description, c.fetchall()))
    return results

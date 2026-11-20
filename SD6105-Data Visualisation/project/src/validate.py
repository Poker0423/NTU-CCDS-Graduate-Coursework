"""Check that the rebuilt outputs still match the numbers quoted in the video and report.

Run after build_dashboard_data.py:  python src/validate.py
"""
from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
claims = pd.read_csv(ROOT / "outputs" / "verified_claims.csv")
cov = pd.read_csv(ROOT / "outputs" / "replay_coverage.csv")
modern = claims[(claims.version == "reviewed") & (claims.period == "2010–2024")].iloc[0]
history = claims[(claims.version == "reviewed") & (claims.period == "1950–2024")].iloc[0]

checks = {
    "305 races, 2010–2024": modern.races == 305,
    "6,436 result rows, 2010–2024": modern.result_rows == 6436,
    "6,351 positive-grid entries, 2010–2024": modern.positive_grid_entries == 6351,
    "P1 won 156 of 305": (modern.pole_wins, modern.pole_entries) == (156, 305),
    "P11+ podiums 38 of 3,307": (modern.p11_podiums, modern.p11_entries) == (38, 3307),
    "historical P11+ podiums 229 of 13,815": (history.p11_podiums, history.p11_entries) == (229, 13815),
    "all six replays have complete lap coverage": bool(cov.coverage_complete.all()) and len(cov) == 6,
}

text = (ROOT / "docs/dashboard/assets/dashboard-data.js").read_text(encoding="utf-8").strip()
payload = json.loads(text[text.index("=") + 1:].rstrip(";"))
sp = next(r for r in payload["datasets"]["reviewed"]["replays"] if r["id"] == "verstappen_2024_sao_paulo")
checks["São Paulo 2024: +16 recorded, cohort rank 14 → 1 (+13)"] = (
    (sp["official_gain"], sp["survivor_start_rank"], sp["survivor_finish_rank"], sp["adjusted_gain"]) == (16, 14, 1, 13))
rev = payload["datasets"]["reviewed"]["records"]
checks["exactly 3 reviewed grid corrections"] = sum(r["g"] != r["rg"] for r in rev) == 3

site = ROOT / "docs"
for html in site.rglob("*.html"):
    for ref in re.findall(r'(?:src|href)="([^"#]+)"', html.read_text(encoding="utf-8")):
        if not ref.startswith(("http:", "https:", "mailto:")):
            checks[f"local link {html.relative_to(ROOT)} -> {ref}"] = (html.parent / ref).is_file()

failed = [k for k, ok in checks.items() if not ok]
for k, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + k)
raise SystemExit(f"{len(failed)} check(s) failed" if failed else 0)

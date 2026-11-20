"""Render the two static figures used in README.md from the dashboard data bundle.

Run after build_dashboard_data.py:  python src/make_readme_figures.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "dashboard" / "assets" / "dashboard-data.js"
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

text = BUNDLE.read_text(encoding="utf-8").strip()
data = json.loads(text[text.index("=") + 1:].rstrip(";"))["datasets"]["reviewed"]
rec = pd.DataFrame(data["records"])
modern = rec[rec.y.between(2010, 2024)]

INK, MUTED, ACCENT, ACCENT2 = "#1f2328", "#6e7781", "#e10600", "#2f6feb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

# 1) Win / podium rate by starting grid, 2010–2024 (all positive-grid entries in the denominator)
g = modern[modern.g.between(1, 20)].groupby("g")
rate = pd.DataFrame({"win": g.f.apply(lambda s: (s == 1).mean()),
                     "podium": g.f.apply(lambda s: s.between(1, 3).mean()),
                     "n": g.size()})
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
ax.bar(rate.index - 0.2, rate.podium * 100, width=0.4, color=ACCENT2, label="Podium (P1–P3)")
ax.bar(rate.index + 0.2, rate.win * 100, width=0.4, color=ACCENT, label="Win")
ax.set_xticks(range(1, 21), [f"P{i}" for i in range(1, 21)], fontsize=8)
ax.set_ylabel("Share of entries (%)")
ax.set_title("Starting grid vs result, Formula 1 2010–2024 (6,351 entries)", loc="left", fontsize=13, color=INK)
ax.legend(frameon=False)
ax.text(1.0, -0.16, "Denominator: every entry with a positive recorded grid, including retirements.",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(IMG / "grid_vs_result.png")
plt.close(fig)

# 2) São Paulo 2024 lap-by-lap position trace for the featured driver
rp = next(r for r in data["replays"] if r["id"] == "verstappen_2024_sao_paulo")
laps = pd.DataFrame(rp["laps"])
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
for d, s in laps.groupby("d"):
    if d != rp["driver_label"]:
        ax.plot(s.l, s.p, color="#d0d7de", lw=1)
s = laps[laps.d == rp["driver_label"]]
ax.plot(s.l, s.p, color=ACCENT, lw=2.6, label=f"{rp['driver_name']} (grid P{rp['grid']} → P{rp['finish']})")
ax.invert_yaxis()
ax.set_yticks([1, 5, 10, 15, 20])
ax.set_xlabel("Lap (0 = starting grid)")
ax.set_ylabel("Position")
ax.set_title("2024 São Paulo Grand Prix: lap-end positions", loc="left", fontsize=13, color=INK)
ax.legend(frameon=False, loc="lower right")
ax.text(1.0, -0.16, "Position changes are not overtake counts; grey traces stop when a driver's lap records end.",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig(IMG / "sao_paulo_2024_trace.png")
plt.close(fig)
print(rate.round(4).head(5).to_string())
print("Saved figures to", IMG)

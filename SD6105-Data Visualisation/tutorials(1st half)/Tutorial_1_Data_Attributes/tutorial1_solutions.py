"""
SD6105 Data Visualisation - Tutorial 1 Solutions (Optimised)
Topic: Data Attributes and Plotting with AI Tools
Datasets: Titanic.csv, hire2FTE.csv

Optimisations from code cheat sheets:
- sns.set_context("talk") for presentation-quality font scaling
- Named matplotlib colours (steelblue, indianred, darkorange, etc.)
- Cleaner annotation using ax.annotate with bbox styling
- df.assign() for derived columns
- Improved figure layout and export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Global style (from Seaborn cheat sheet) ───────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.9)   # Better font scaling for presentations

DATA = "./data/"
OUT  = "./charts/"

# =============================================================================
# SECTION 1.1: DATA ATTRIBUTES — WRITTEN ANSWERS
# =============================================================================
print("=" * 60)
print("SECTION 1.1: DATA ATTRIBUTES")
print("=" * 60)
print("""
Q1 – Continuous vs Discrete:
  (a) No. of patients in ward          → DISCRETE  (counted, integers)
  (b) Blood pressure                   → CONTINUOUS (measured)
  (c) Pulse rate (counted beats/min)   → DISCRETE  (integer count)
      Pulse rate (sensor waveform)     → CONTINUOUS (measured signal)
  (d) ER wait time (rounded, lobby)    → DISCRETE  (integer minutes)
  (e) ER wait time per patient(rounded)→ DISCRETE  (integer minutes)

Q2 – NOIR Scale:
  (a) Rank in Singapore Army   → ORDINAL  (Private < Cpl < Sgt …)
  (b) No. of students (CZ4124) → RATIO    (absolute zero, can divide)
  (c) Shoe sizes               → ORDINAL  (ordered, unequal intervals)
  (d) Jersey numbers           → NOMINAL  (labels only, no order)

Q3 – Titanic NOIR:
  PassengerId → NOMINAL  | Survived → NOMINAL  | Pclass  → ORDINAL
  Name        → NOMINAL  | Sex      → NOMINAL  | Age     → RATIO
  SibSp       → RATIO    | Parch    → RATIO    | Ticket  → NOMINAL
  Fare        → RATIO    | Cabin    → NOMINAL  | Embarked→ NOMINAL

Q4 – Nielsen "Mobile OS by Age" clustered bar:
  3 data dimensions:
    ① Age group   (Ordinal: 18-24, 25-34, 35-54, 55+)
    ② Mobile OS   (Nominal: iOS, Android, Other)
    ③ Market share (Ratio: 0-100%)
""")

# =============================================================================
# SECTION 1.2 Q1: Inspect hire2FTE dataset
# =============================================================================
print("=" * 60)
print("SECTION 1.2: CREATING PLOTS")
print("=" * 60)

df_raw = pd.read_csv(DATA + "hire2FTE.csv", index_col=0)
print("\nhire2FTE dataset:")
print(df_raw.to_string())

months   = df_raw.columns.tolist()
received = df_raw.loc["Received"].astype(int).values
processed = df_raw.loc["Processed"].astype(int).values
shortfall = received - processed
x = np.arange(len(months))

# =============================================================================
# Q2: Cluster Bar Chart with annotations
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))
W = 0.35

b1 = ax.bar(x - W/2, received,  W, label="Received",  color="steelblue",  alpha=0.9)
b2 = ax.bar(x + W/2, processed, W, label="Processed", color="darkorange", alpha=0.9)

# Value labels on bars (from Seaborn cheat sheet pattern — iterate bars)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{int(bar.get_height())}", ha="center", va="bottom",
            fontsize=7.5, color="dimgrey")

# Annotation: 2 workers quit in May (index 4)
ax.annotate("2 workers quit",
            xy=(4, max(received[4], processed[4]) + 5),
            xytext=(5.2, max(received[4], processed[4]) + 28),
            arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.8),
            fontsize=10, color="firebrick",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="firebrick", alpha=0.9))

ax.set_xticks(x); ax.set_xticklabels(months)
ax.set_xlabel("Month"); ax.set_ylabel("Transaction Volume")
ax.set_title("Monthly Transaction Volume: Received vs Processed\n"
             "(Cluster bar — hard to perceive growing gap)", fontweight="bold")
ax.legend()
ax.set_ylim(0, max(received) * 1.35)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT + "q2_cluster_bar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Q2 chart saved")

print("""
Q3 – Why cluster bar is NOT effective:
  The viewer must mentally SUBTRACT adjacent bar heights to perceive
  the shortfall. This violates Weber's Law — the small gap (~5-30 units)
  is imperceptible against the ~140-240 unit bar heights. The growing
  deficit — the key message — is completely hidden in side-by-side format.
""")

# =============================================================================
# Q4: Effective Line Chart (fill_between shading for shortfall)
# =============================================================================
# Colours chosen from Matplotlib colour names cheat sheet
BLUE   = "steelblue"
ORANGE = "darkorange"
RED    = "firebrick"

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(months, received,  "o-", color=BLUE,   lw=2.5, ms=7, label="Received",  zorder=3)
ax.plot(months, processed, "s-", color=ORANGE, lw=2.5, ms=7, label="Processed", zorder=3)

# Shade growing shortfall
ax.fill_between(months, received, processed,
                where=(received > processed),
                alpha=0.22, color=RED, label="Unprocessed Shortfall")

# Vertical dashed line at May
ax.axvline(x=4, color=RED, ls="--", lw=1.6, alpha=0.75)
ax.text(4.12, max(received) * 0.93,
        "← 2 workers quit\n   (May)",
        color=RED, fontsize=9.5, va="top")

# Annotate shortfall numerics from June onward
for i in range(5, len(months)):
    gap = int(received[i] - processed[i])
    if gap > 0:
        mid = (received[i] + processed[i]) / 2
        ax.text(i, mid, f"−{gap}", ha="center", va="center",
                fontsize=8.5, color="darkred", fontweight="bold")

ax.set_xlabel("Month (2023)"); ax.set_ylabel("Transaction Volume")
ax.set_title("Growing Processing Backlog After Staff Reduction in May\n"
             "→ Line chart with shading makes the shortfall immediately visible",
             fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, max(received) * 1.28)
ax.set_axisbelow(True)

# Highlight problem months on x-axis (red tick labels from May onward)
for i, lbl in enumerate(ax.get_xticklabels()):
    if i >= 4:
        lbl.set_color(RED)

plt.tight_layout()
plt.savefig(OUT + "q4_line_chart_effective.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Q4 chart saved")

print("""
Q4 Commentary – Visual effectiveness:
  COLOURS:     Blue (received) / Orange (processed) — colourblind-friendly,
               high contrast pair. Named from Matplotlib colour cheat sheet.
  MARKERS:     Circle vs square → redundant encoding (works in B&W print too)
  SHADING:     Red fill immediately shows WHERE and HOW MUCH the gap exists.
               The absence of shading before May reinforces the before/after story.
  ANNOTATION:  Red vertical dashed line + text pinpoints the causal event.
  HIGHLIGHTS:  Numeric labels inside the gap quantify exact monthly shortfall.
""")

# =============================================================================
# Q5: Combined Line + Bar (dual-axis)  — uses df.assign() pattern
# =============================================================================
# Build a tidy DataFrame first (df.assign from Pandas cheat sheet)
monthly = pd.DataFrame({
    "month":     months,
    "received":  received,
    "processed": processed
}).assign(shortfall=lambda d: d.received - d.processed)

fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

# Shortfall bars on secondary axis
bar_colors = [RED if s > 0 else "lightsteelblue" for s in monthly.shortfall]
ax2.bar(monthly.month, monthly.shortfall, color=bar_colors, alpha=0.38,
        label="Shortfall (Received − Processed)", zorder=1)
ax2.axhline(0, color="grey", lw=0.8, ls="--")
ax2.set_ylabel("Shortfall (units)", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")

# Lines on primary axis
ax1.plot(monthly.month, monthly.received,  "o-", color=BLUE,   lw=2.5, ms=7,
         label="Received", zorder=3)
ax1.plot(monthly.month, monthly.processed, "s-", color=ORANGE, lw=2.5, ms=7,
         label="Processed", zorder=3)
ax1.axvline(x=4, color=RED, ls="--", lw=1.6, alpha=0.8)
ax1.text(4.12, max(received) * 0.90, "← Staff cuts\n   (May)",
         color=RED, fontsize=9.5)

ax1.set_xlabel("Month (2023)"); ax1.set_ylabel("Transaction Volume")
ax1.set_title("Combined Line + Bar Chart: Volume & Monthly Shortfall\n"
              "Growing red bars prove the need for 2 additional FTEs",
              fontweight="bold")

lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbl1 + lbl2, fontsize=10, loc="upper left")
ax1.yaxis.grid(True, alpha=0.2); ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT + "q5_combined_line_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Q5 chart saved")

print("\n✅ Tutorial 1 complete — all charts saved to completed_tutorials/Tutorial_1/")

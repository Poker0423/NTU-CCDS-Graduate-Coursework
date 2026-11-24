"""
SD6105 Data Visualisation - Tutorial 5 Solutions
Topic: Tableau - basics, maps and interactive dashboards (Sample - Superstore)

Tutorial 5 is done in Tableau Desktop (GUI). The step-by-step Tableau recipe for
every exercise is in the README. This script REPRODUCES each Tableau worksheet in
Python on the same data so that the results can be shown on GitHub, and it
generates `dashboard.html`, an interactive D3 version of the Part-3 dashboard
with the same three practices:
  Practice 1  navigation button      (Overview <-> Details view)
  Practice 2  dashboard filter action (click a bar or a state to filter the rest)
  Practice 3  URL actions             (fixed web address + Wikipedia page of the
                                       selected state, dynamic)

Run from inside this folder:  python tutorial5_solutions.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

os.makedirs("charts", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
TABLEAU_BLUE = "#4e79a7"
CAT_COL = {"Furniture": "#4e79a7", "Office Supplies": "#f28e2b", "Technology": "#e15759"}
usd_k = mtick.FuncFormatter(lambda v, _: f"${v/1e3:,.0f}K")

orders = pd.read_excel("data/Sample - Superstore.xls", sheet_name="Orders")
orders["Month"] = orders["Order Date"].dt.to_period("M").dt.to_timestamp()
print(f"Orders: {len(orders):,} rows, {orders['Order Date'].min():%b %Y} - "
      f"{orders['Order Date'].max():%b %Y}, total sales ${orders.Sales.sum():,.0f}")

# =============================================================================
# PART 1.1  Basic bar chart  (Figure 1.12: SUM(Sales) by Category)
#   Tableau: drag Category -> Columns, Sales -> Rows; drag Category -> Color.
# =============================================================================
cat = orders.groupby("Category")["Sales"].sum()
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(cat.index, cat.values, color=[CAT_COL[c] for c in cat.index], width=0.6)
ax.bar_label(bars, labels=[f"${v/1e3:,.0f}K" for v in cat.values], padding=3)
ax.yaxis.set_major_formatter(usd_k)
ax.set_ylabel("Sales")
ax.set_title("Part 1 - Sales by Category (Fig 1.12)", loc="left", weight="bold")
fig.tight_layout()
fig.savefig("charts/q1_bar_sales_by_category.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# PART 1.2  Line chart - sales across years and months
#   Tableau: Order Date -> Columns, right-click -> MONTH (continuous, green pill);
#            Sales -> Rows.  (Discrete YEAR + MONTH gives the "yearly panels" view.)
# =============================================================================
monthly = orders.groupby("Month")["Sales"].sum()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5))
ax1.plot(monthly.index, monthly.values, color=TABLEAU_BLUE, lw=1.8, marker="o", ms=3)
ax1.yaxis.set_major_formatter(usd_k)
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.set_title("Continuous MONTH(Order Date) - strong seasonality, peaks every Sep/Nov/Dec",
              loc="left", weight="bold")
by_year = orders.assign(Y=orders["Order Date"].dt.year, M=orders["Order Date"].dt.month) \
                .pivot_table(index="M", columns="Y", values="Sales", aggfunc="sum")
greys = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, by_year.shape[1]))
for (y, s), c in zip(by_year.items(), greys):
    ax2.plot(s.index, s.values, color=c, lw=2, label=str(y))
    ax2.text(12.15, s.iloc[-1], str(y), color=c, va="center", fontsize=9)
ax2.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                              "Sep", "Oct", "Nov", "Dec"])
ax2.yaxis.set_major_formatter(usd_k)
ax2.set_title("Discrete MONTH on Columns, YEAR on Color - years compared month by month",
              loc="left", weight="bold")
fig.suptitle("Part 1 - Sales across years and months", weight="bold", x=0.01, ha="left")
fig.tight_layout()
fig.savefig("charts/q2_line_sales_by_month.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# PART 1.3  Multiple bar charts - Sales by Region and Category
#   Tableau: Region -> Columns, Category -> Rows, Sales -> Columns (horizontal bars).
# =============================================================================
rc = orders.pivot_table(index="Category", columns="Region", values="Sales", aggfunc="sum")
regions = ["Central", "East", "South", "West"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, sharex=True)
for ax, r in zip(axes, regions):
    v = rc[r].reindex(list(CAT_COL)[::-1])
    b = ax.barh(v.index, v.values, color=TABLEAU_BLUE, height=0.6)
    ax.bar_label(b, labels=[f"${x/1e3:,.0f}K" for x in v.values], padding=3, fontsize=8)
    ax.set_title(r, weight="bold")
    ax.xaxis.set_major_formatter(usd_k)
    ax.set_xlim(0, rc.values.max() * 1.3)
fig.suptitle("Part 1 - Sales by Region (columns) and Category (rows)", weight="bold",
             x=0.01, ha="left")
fig.tight_layout()
fig.savefig("charts/q3_bars_sales_region_category.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# PART 2  Symbol map (Figure 6.1)
#   Tableau: double-click Country, State, City (geographic roles -> Latitude/
#   Longitude (generated)); Sales -> Size (and Color).
#   Offline Python has no basemap/geocoder, so the same idea is shown as a
#   proportional-symbol TILE-GRID MAP by state (circle area ~ sales), with the
#   top cities listed - the HTML dashboard uses the same layout.
# =============================================================================
TILES = {  # state: (row, col) - common US tile-grid layout
    "Maine": (0, 11), "Vermont": (1, 10), "New Hampshire": (1, 11),
    "Washington": (2, 1), "Idaho": (2, 2), "Montana": (2, 3), "North Dakota": (2, 4),
    "Minnesota": (2, 5), "Illinois": (2, 6), "Wisconsin": (2, 7), "Michigan": (2, 8),
    "New York": (2, 9), "Rhode Island": (2, 10), "Massachusetts": (2, 11),
    "Oregon": (3, 1), "Nevada": (3, 2), "Wyoming": (3, 3), "South Dakota": (3, 4),
    "Iowa": (3, 5), "Indiana": (3, 6), "Ohio": (3, 7), "Pennsylvania": (3, 8),
    "New Jersey": (3, 9), "Connecticut": (3, 10),
    "California": (4, 1), "Utah": (4, 2), "Colorado": (4, 3), "Nebraska": (4, 4),
    "Missouri": (4, 5), "Kentucky": (4, 6), "West Virginia": (4, 7), "Virginia": (4, 8),
    "Maryland": (4, 9), "Delaware": (4, 10),
    "Arizona": (5, 2), "New Mexico": (5, 3), "Kansas": (5, 4), "Arkansas": (5, 5),
    "Tennessee": (5, 6), "North Carolina": (5, 7), "South Carolina": (5, 8),
    "District of Columbia": (5, 9),
    "Oklahoma": (6, 4), "Louisiana": (6, 5), "Mississippi": (6, 6), "Alabama": (6, 7),
    "Georgia": (6, 8), "Texas": (7, 4), "Florida": (7, 9),
    "Alaska": (0, 0), "Hawaii": (7, 0),
}
ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
        "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
        "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
        "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
state_sales = orders.groupby("State")["Sales"].sum()
state_region = orders.groupby("State")["Region"].first()
REG_COL = {"Central": "#76b7b2", "East": "#59a14f", "South": "#edc948", "West": "#b07aa1"}

fig, ax = plt.subplots(figsize=(13, 8))
smax = state_sales.max()
for st, (r, c) in TILES.items():
    ax.add_patch(plt.Rectangle((c - 0.47, r - 0.47), 0.94, 0.94, fc="#f3f3f3", ec="white"))
    if st in state_sales:
        v = state_sales[st]
        rad = 0.46 * np.sqrt(v / smax)              # circle AREA proportional to sales
        ax.add_patch(plt.Circle((c, r), max(rad, 0.04), color=REG_COL[state_region[st]],
                                alpha=0.85))
        ax.text(c, r - 0.36, ABBR[st], ha="center", va="center", fontsize=8, weight="bold")
    else:
        ax.text(c, r, ABBR[st], ha="center", va="center", fontsize=8, color="#bbbbbb")
top_cities = orders.groupby(["City", "State"])["Sales"].sum().nlargest(8)
txt = "Top cities by sales\n" + "\n".join(f"{c}, {ABBR[s]}: ${v/1e3:,.0f}K"
                                          for (c, s), v in top_cities.items())
ax.text(12.0, 4.4, txt, fontsize=9, va="top", family="monospace")
for i, (reg, col) in enumerate(REG_COL.items()):
    ax.add_patch(plt.Circle((0.2 + i * 1.5, 8.4), 0.15, color=col))
    ax.text(0.45 + i * 1.5, 8.4, reg, va="center", fontsize=9)
for j, v in enumerate([25_000, 100_000, 400_000]):
    ax.add_patch(plt.Circle((7.0 + j * 1.3, 8.4), 0.46 * np.sqrt(v / smax), fill=False, ec="grey"))
    ax.text(7.0 + j * 1.3, 8.95, f"${v/1e3:,.0f}K", ha="center", fontsize=8, color="grey")
ax.set_xlim(-0.6, 15.2)
ax.set_ylim(9.2, -0.6)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Part 2 - Symbol map: total sales by state (circle area = sales, colour = region)\n"
             "states without orders (AK, HI) are grey", loc="left", weight="bold")
fig.tight_layout()
fig.savefig("charts/q4_symbol_map_sales_by_state.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# PART 3  Interactive dashboard (Figure 8.6) -> dashboard.html
#   Aggregate once in pandas and embed as JSON so the page also works when opened
#   directly from disk (no web server needed).
# =============================================================================
agg = orders.groupby(["State", "Region", "Category", "Month"], as_index=False) \
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
months = sorted(agg["Month"].unique())
m_idx = {m: i for i, m in enumerate(months)}
city = orders.groupby(["State", "City"], as_index=False).agg(Sales=("Sales", "sum"),
                                                             Profit=("Profit", "sum"))
payload = {
    "months": [pd.Timestamp(m).strftime("%Y-%m") for m in months],
    "rows": [[r.State, r.Region, r.Category, m_idx[r.Month], round(r.Sales, 2),
              round(r.Profit, 2)] for r in agg.itertuples()],
    "cities": [[r.State, r.City, round(r.Sales, 2), round(r.Profit, 2)]
               for r in city.itertuples()],
    "tiles": {s: [r, c, ABBR[s]] for s, (r, c) in TILES.items()},
}
with open("dashboard_template.html", encoding="utf-8") as f:
    html = f.read().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
print(f"dashboard.html written ({len(html)/1e3:.0f} KB) - open it in a browser")
print("Tutorial 5 charts saved to ./charts/")

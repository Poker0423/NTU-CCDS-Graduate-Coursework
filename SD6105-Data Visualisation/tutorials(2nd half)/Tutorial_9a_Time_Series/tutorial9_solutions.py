"""
SD6105 Data Visualisation - Tutorial 9 Solutions
Topic: Time-series Data Visualization

9.1 (a) Music trends  - 4 genres, daily play counts, 5 years     -> STREAMGRAPH
9.1 (b) Server load    - 100 servers, CPU per minute, 30 days     -> HORIZON GRAPHS
                                                                    (+ sorted heatmap)
9.1 (c) Website traffic - hourly traffic for 1 year               -> CALENDAR HEATMAP
                                                                    (+ weekday x hour view)

No data files are provided with the handout, so each scenario is demonstrated
with SYNTHETIC data generated with a fixed seed (charts in ./charts/).

Run from inside this folder:  python tutorial9_solutions.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

os.makedirs("charts", exist_ok=True)
rng = np.random.default_rng(9)
sns.set_theme(style="white")

# =============================================================================
# 9.1 (a)  MUSIC TRENDS  ->  STREAMGRAPH (ThemeRiver)
# =============================================================================
"""
Proposed design: STREAMGRAPH - a stacked area chart with a wiggle-minimising,
centred baseline and smooth (weekly-aggregated) curves; one colour per genre,
genres labelled directly inside their stream. Hover tooltip with the exact
daily/weekly value, plus an optional toggle to a 100% stacked area view.

Justification against the three goals
 1. Evolution of each genre: each stream's THICKNESS at time t = that genre's
    play count, so growth/decline of a genre reads as the stream swelling or
    thinning along the time axis.
 2. Relative proportions: streams are stacked, so the share of each genre at
    any time is the fraction of the total thickness; the 100%-normalised
    variant makes the proportions exact.
 3. Overall "flow": the organic, centred shape emphasises the continuous
    evolution of the whole mix and the total volume (overall thickness), which
    is exactly the ThemeRiver metaphor. Only 4 categories -> no clutter.
 Trade-offs: absolute values of individual streams are harder to read than in
 a line chart (no common baseline) -> keep tooltips, or show small-multiple
 line charts as a secondary view. Daily data are noisy -> aggregate to weeks.
"""
days = pd.date_range("2021-01-01", periods=5 * 365, freq="D")
t = np.arange(len(days)) / 365
season = 1 + 0.10 * np.sin(2 * np.pi * (t - 0.9))          # holiday peak
genres = {
    "Pop":        12000 * (1 + 0.15 * t) * season,
    "Rock":       9000 * np.exp(-0.18 * t) * season,
    "Electronic": 3000 * (1 + 0.9 * t ** 1.4) * (1 + 0.15 * np.sin(2 * np.pi * (t - 0.4))),
    "Classical":  2500 * (1 + 0.05 * np.sin(2 * np.pi * t)) * (1 + 0.04 * t),
}
music = pd.DataFrame({g: v * rng.lognormal(0, 0.08, len(days)) for g, v in genres.items()},
                     index=days)
weekly = music.resample("W").sum()
gcol = {"Pop": "#e7298a", "Rock": "#7570b3", "Electronic": "#1b9e77", "Classical": "#d95f02"}

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
ys = ax.stackplot(weekly.index, weekly.T.values / 1e3, baseline="wiggle",
                  colors=list(gcol.values()), alpha=0.9, edgecolor="white", linewidth=0.3)
# direct labels at the widest point of each stream
for poly, g in zip(ys, gcol):
    verts = poly.get_paths()[0].vertices
    k = np.argmax(weekly[g].values)
    x = mdates.date2num(weekly.index[k])
    yv = verts[np.abs(verts[:, 0] - x) < 4, 1]
    if len(yv):
        thin = weekly[g].max() < 0.15 * weekly.sum(axis=1).max()
        ax.text(weekly.index[k], yv.max() + (12 if thin else 0) if thin else yv.mean(), g,
                ha="center", va="bottom" if thin else "center",
                color=gcol[g] if thin else "white", weight="bold", fontsize=11)
ax.set_yticks([])
ax.set_title("9.1(a) Streamgraph - weekly plays by genre (thickness = plays)",
             loc="left", weight="bold")
sns.despine(ax=ax, left=True)

share = weekly.div(weekly.sum(axis=1), axis=0) * 100
ax2.stackplot(share.index, share.T.values, colors=list(gcol.values()), alpha=0.9)
ax2.set_ylim(0, 100)
ax2.set_ylabel("% of plays")
ax2.set_title("Companion view: 100% stacked area for exact proportions", loc="left")
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
sns.despine(ax=ax2)
fig.text(0.01, 0.005, "Synthetic data", fontsize=8, color="grey")
fig.tight_layout()
fig.savefig("charts/q1a_music_streamgraph.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 9.1 (b)  SERVER LOAD  ->  HORIZON GRAPHS (+ similarity-sorted heatmap)
# =============================================================================
"""
Proposed design: one HORIZON GRAPH row per server (100 rows stacked
vertically, ~20-25 px each), aligned on a shared time axis, rows ordered by
similarity (hierarchical clustering / correlation of their load profiles),
with interactive brushing/zoom on the time axis and a detail line chart on
click.

Justification
 1. Detect spikes quickly: the horizon graph divides the value range into
    bands (e.g. 0-33-66-100 % CPU) that are layered with increasingly dark
    colour; high load = dark red, so spikes pop out pre-attentively even in a
    very short row. Deviations below a baseline can be mirrored in blue.
 2. 100 series in limited space: layering + mirroring cuts the vertical space
    of each series by 2-4x while keeping value resolution (Heer et al. show
    horizon graphs remain accurate at small heights where line charts fail).
 3. Similarities across servers: rows share one time axis; after sorting by
    similarity, servers with the same daily pattern form visual blocks, and
    simultaneous spikes line up vertically (e.g. a fleet-wide incident).
 Alternative/companion: a heatmap (servers x time, colour = CPU) is even more
 compact for an overview of all 100 servers; the horizon graph adds better
 magnitude reading. Per-minute data (43,200 points/server) should be
 aggregated (e.g. 10-min max) for display, with zoom for detail.
"""
n_srv, minutes = 100, 30 * 24 * 60
tm = pd.date_range("2026-10-01", periods=minutes, freq="min")
hours = (np.arange(minutes) / 60) % 24
profiles = {
    "web":   35 + 30 * np.exp(-((hours - 14) / 4) ** 2),     # daytime peak
    "batch": 25 + 50 * np.exp(-((hours - 2) / 1.5) ** 2),    # nightly jobs
    "db":    45 + 10 * np.sin(2 * np.pi * hours / 24),
    "idle":  np.full(minutes, 12.0),
}
kind = rng.choice(list(profiles), n_srv, p=[0.4, 0.25, 0.25, 0.1])
cpu = np.empty((n_srv, minutes), dtype=np.float32)
for i, k in enumerate(kind):
    s = profiles[k] * rng.uniform(0.85, 1.15) + rng.normal(0, 4, minutes)
    for _ in range(rng.integers(0, 4)):                      # random spikes
        st = rng.integers(0, minutes - 120)
        s[st:st + rng.integers(10, 120)] += rng.uniform(30, 55)
    cpu[i] = np.clip(s, 0, 100)
inc = int(17.4 * 24 * 60)                                    # fleet-wide incident
cpu[:, inc:inc + 90] = np.clip(cpu[:, inc:inc + 90] + 40, 0, 100)

# aggregate to 30-min max for display
agg = cpu.reshape(n_srv, -1, 30).max(axis=2)
t_agg = tm[::30]
# order servers by similarity (correlation-based greedy ordering)
corr = np.corrcoef(agg)
order = [int(np.argmax(agg.mean(1)))]
remaining = set(range(n_srv)) - set(order)
while remaining:
    nxt = max(remaining, key=lambda j: corr[order[-1], j])
    order.append(nxt)
    remaining.remove(nxt)
agg_sorted = agg[order]


def horizon(ax, x, y, bands=3, vmax=100, cmap="Reds"):
    """Draw a (positive-only) horizon graph in ax."""
    cols = plt.get_cmap(cmap)(np.linspace(0.35, 0.95, bands))
    band_h = vmax / bands
    for b in range(bands):
        yb = np.clip(y - b * band_h, 0, band_h)
        ax.fill_between(x, 0, yb, color=cols[b], linewidth=0, step="mid")
    ax.set_ylim(0, band_h)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


n_show = 25
fig, axes = plt.subplots(n_show, 1, figsize=(13, 9), sharex=True)
for r, ax in enumerate(axes):
    i = order[r * (n_srv // n_show)]
    horizon(ax, t_agg, agg[i])
    ax.text(-0.005, 0.5, f"srv-{i:03d} ({kind[i]})", transform=ax.transAxes,
            ha="right", va="center", fontsize=7)
axes[-1].xaxis.set_visible(True)
axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[-1].tick_params(axis="x", labelsize=8)
fig.suptitle("9.1(b) Horizon graphs - CPU load (30-min max), 3 bands: 0-33 / 33-67 / 67-100 %\n"
             "every 4th of 100 servers shown, ordered by similarity; dark red = spike",
             weight="bold", fontsize=11)
fig.subplots_adjust(hspace=0.12, left=0.12)
fig.savefig("charts/q1b_server_horizon_graphs.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 6.5))
im = ax.imshow(agg_sorted, aspect="auto", cmap="magma_r", vmin=0, vmax=100,
               extent=[mdates.date2num(t_agg[0]), mdates.date2num(t_agg[-1]), n_srv, 0])
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_ylabel("Server (ordered by similarity)")
ax.annotate("fleet-wide incident", (mdates.date2num(tm[inc]), 0), xytext=(0, 12),
            textcoords="offset points", ha="center", fontsize=9, color="#b2182b",
            arrowprops=dict(arrowstyle="->", color="#b2182b"))
fig.colorbar(im, ax=ax, label="CPU % (30-min max)", pad=0.01)
ax.set_title("9.1(b) Companion overview: all 100 servers as a heatmap", loc="left", weight="bold")
fig.tight_layout()
fig.savefig("charts/q1b_server_heatmap.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 9.1 (c)  WEBSITE TRAFFIC  ->  CALENDAR HEATMAP (+ weekday x hour heatmap)
# =============================================================================
"""
Proposed design: CALENDAR HEATMAP (calendar view) - 12 month blocks, each a
grid of weeks (columns) x weekdays (rows); each cell = one date coloured by
its average hourly traffic with a sequential palette; tooltips with exact
values. Companion view: a weekday x hour-of-day heatmap (cycle view).

Justification
 1. Variation across dates: every date of the year has its own cell; a
    sequential colour scale makes high/low days (news events, holidays)
    stand out immediately.
 2. Weekday vs weekend: weekdays are aligned in ROWS, so a lighter/darker
    Saturday-Sunday band is visible across the whole year; the weekday x hour
    heatmap shows HOW the daily cycle differs (e.g. weekday commute peaks).
 3. Compare months: months are separate, identically structured blocks placed
    in order, so their overall colour can be compared side by side; a small
    monthly-mean bar strip can be added for exact comparison.
 Hourly granularity is aggregated into the daily average for the calendar
 (fits 365 cells on one screen); the hourly pattern is preserved in the
 companion view (and could be a drill-down on click).
"""
hrs = pd.date_range("2026-01-01", "2026-12-31 23:00", freq="h")
h = hrs.hour.values
dow = hrs.dayofweek.values
doy = hrs.dayofyear.values
daily_shape = 400 + 900 * np.exp(-((h - 9) / 2.5) ** 2) + 700 * np.exp(-((h - 20) / 2.5) ** 2)
weekend = np.where(dow >= 5, 0.65, 1.0)
seasonal = 1 + 0.15 * np.cos(2 * np.pi * (doy - 20) / 365)
traffic = daily_shape * weekend * seasonal * rng.lognormal(0, 0.12, len(hrs))
for d in rng.choice(np.arange(1, 366), 6, replace=False):          # breaking-news days
    traffic[doy == d] *= rng.uniform(1.8, 2.6)
web = pd.DataFrame({"traffic": traffic}, index=hrs)
daily = web["traffic"].resample("D").mean()

fig, axes = plt.subplots(3, 4, figsize=(15, 8.5))
vmin, vmax = daily.min(), daily.quantile(0.99)
cmap = plt.get_cmap("YlGnBu")
for m, ax in enumerate(axes.flat, start=1):
    d = daily[daily.index.month == m]
    first_dow = d.index[0].dayofweek
    grid = np.full((7, 6), np.nan)
    for date, v in d.items():
        k = date.day - 1 + first_dow
        grid[date.dayofweek, k // 7] = v
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    for date in d.index:
        k = date.day - 1 + first_dow
        ax.text(k // 7, date.dayofweek, date.day, ha="center", va="center", fontsize=6,
                color="white" if d[date] > (vmin + vmax) / 2 else "#333333")
    ax.set_title(f"{d.index[0]:%B}  (mean {d.mean():,.0f})", fontsize=10)
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=7)
    ax.set_xticks([])
    ax.axhline(4.5, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
fig.colorbar(sm, ax=axes, shrink=0.6, label="Average hourly visits (daily mean)")
fig.suptitle("9.1(c) Calendar heatmap - website traffic 2026 (synthetic)", weight="bold", fontsize=13)
fig.savefig("charts/q1c_traffic_calendar_heatmap.png", bbox_inches="tight")
plt.close(fig)

pivot = web.assign(dow=web.index.day_name().str[:3], hour=web.index.hour) \
           .pivot_table(index="dow", columns="hour", values="traffic", aggfunc="mean") \
           .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
fig, ax = plt.subplots(figsize=(13, 3.8))
sns.heatmap(pivot, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Avg visits / hour"})
ax.set_xlabel("Hour of day")
ax.set_ylabel("")
ax.axhline(5, color="white", lw=3)
ax.set_title("9.1(c) Companion view: weekday x hour - weekend traffic is ~35% lower, "
             "morning & evening peaks", loc="left", weight="bold")
fig.tight_layout()
fig.savefig("charts/q1c_traffic_weekday_hour.png", bbox_inches="tight")
plt.close(fig)

print("Tutorial 9 charts saved to ./charts/")

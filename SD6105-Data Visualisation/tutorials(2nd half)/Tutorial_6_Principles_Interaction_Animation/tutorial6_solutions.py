"""
SD6105 Data Visualisation - Tutorial 6 Solutions
Topic: Visualization Principles, Interaction and Animation

6.1  Critique + redesign of the "Average Home Prices - February (GTA)" bar chart
6.2  Critique + redesign of the "Tech Stock Throwback?" dual-axis line chart
6.3  Interaction taxonomy (Yi et al., 2007) applied to the Compendia system
6.4  Congruence vs. Apprehension principles of animation

Run from inside this folder:  python tutorial6_solutions.py
Charts are written to ./charts/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

os.makedirs("charts", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})

# =============================================================================
# 6.1  AVERAGE HOME PRICES IN FEBRUARY (GREATER TORONTO AREA)
# =============================================================================
# Values are read off the newspaper chart's gridlines (y-axis starts at $775,000,
# one gridline = $75,000), so they are approximate (+/- $5K). Units sold are
# printed on the chart itself. Source on the chart: TRREB.
years = np.array([2020, 2021, 2022, 2023])
avg_price = np.array([905_000, 1_035_000, 1_335_000, 1_100_000])   # approx.
units_sold = np.array([7_193, 10_929, 9_028, 4_783])                # exact (printed)

"""
6.1 (a) Two major issues

ISSUE 1 - Truncated value axis (violates GRAPHICAL INTEGRITY).
  The y-axis starts at $775,000 instead of 0, but bars encode value by LENGTH.
  Bar length is therefore proportional to (price - 775K), not to price.
  The 2022 bar looks ~4x as tall as the 2020 bar, although the price is only
  ~1.5x higher.  Tufte's Lie Factor (computed below) is ~7, far above the
  acceptable 0.95-1.05 band.  Bars (area/length marks) must start at zero.

ISSUE 2 - Confusing / missing encoding of the second variable (units sold) and
  poor labelling.
  * "Units Sold" (7,193 / 10,929 / 9,028 / 4,783) is printed as the x-axis
    category labels, while the real category (year) is written inside the bars.
    A reader naturally reads the x-axis as the category -> misreading.
  * Units sold is a second quantitative variable that is NOT visually encoded
    at all (only text), so the interesting story - prices fell in 2023 while
    sales volume collapsed - cannot be seen.
  * The exact price of each bar is not labelled; readers must interpolate
    between 9 dense gridlines with odd step values ($75K), on a grey textured
    background with a heavy frame (low data-ink ratio).

6.1 (b) Alternative design - two aligned panels sharing the year axis
  Panel 1: average price as bars that START AT ZERO, each bar directly labelled
           with its value and the year-over-year % change.
           -> fixes Issue 1 (Lie Factor = 1) and removes gridline interpolation.
  Panel 2: units sold as a separate bar chart aligned on the same x-axis (year).
           -> fixes Issue 2: each quantitative variable gets its own position/
              length channel; year is the shared, clearly-labelled category.
  Advantages
   - Truthful magnitudes; % change annotations still make the differences
     salient without exaggerating them.
   - Small multiples with a common x-axis invite comparison of price vs. volume
     (2022 peak price, 2023 drop in both).
   - Light styling, no background texture, minimal gridlines (higher data-ink).
  (An alternative is a connected scatterplot of price vs. units sold with the
   year labelled on each point - good for showing the joint trajectory, but less
   familiar for a newspaper audience.)

6.1 (c) Sketch -> charts/q1c_redesign_sketch.png  (produced by the code below)
"""


def lie_factor(values, baseline):
    """Tufte's lie factor between the smallest and largest value."""
    lo, hi = values.min(), values.max()
    effect_data = (hi - lo) / lo
    effect_graphic = ((hi - baseline) - (lo - baseline)) / (lo - baseline)
    return effect_graphic / effect_data


lf = lie_factor(avg_price, 775_000)
print(f"6.1  Lie factor of the original chart (baseline $775K): {lf:.1f}")

# ---- Original (re-created) vs. honest baseline -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.bar(years.astype(str), avg_price, color="#b2182b", width=0.6)
ax.set_ylim(775_000, 1_375_000)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.set_xticks(range(4), [f"{u:,}" for u in units_sold])
ax.set_xlabel("Units Sold  (years written inside bars in the original)")
for i, y in enumerate(years):
    ax.text(i, avg_price[i] - 40_000, str(y), ha="center", color="white", weight="bold")
ax.set_title(f"Original design (re-created)\ny-axis starts at $775K  ->  Lie Factor ≈ {lf:.1f}",
             fontsize=11, color="#b2182b")

ax = axes[1]
ax.bar(years.astype(str), avg_price, color="#888888", width=0.6)
ax.set_ylim(0, 1_450_000)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.set_title("Same data, zero baseline\n2022 is ~1.5x 2020, not ~4x", fontsize=11)
fig.suptitle("6.1(a)  Truncated axis exaggerates differences", weight="bold")
fig.tight_layout()
fig.savefig("charts/q1a_truncated_axis_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---- Proposed redesign ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1]})
c_price, c_units = "#2166ac", "#9e9e9e"
bars = ax1.bar(years, avg_price, color=c_price, width=0.6)
ax1.set_ylim(0, 1_600_000)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax1.set_ylabel("Average price")
for i, (b, v) in enumerate(zip(bars, avg_price)):
    ax1.text(b.get_x() + b.get_width() / 2, v + 30_000, f"${v/1e3:,.0f}K",
             ha="center", va="bottom", weight="bold", color=c_price)
    if i:
        pct = (v - avg_price[i - 1]) / avg_price[i - 1] * 100
        ax1.text(b.get_x() + b.get_width() / 2, v / 2, f"{pct:+.0f}%\nvs {years[i-1]}",
                 ha="center", va="center", color="white", fontsize=10)
ax1.set_title("Average home price, February (bars start at $0)", loc="left", fontsize=11)

bars2 = ax2.bar(years, units_sold, color=c_units, width=0.6)
for b, v in zip(bars2, units_sold):
    ax2.text(b.get_x() + b.get_width() / 2, v + 250, f"{v:,}", ha="center", va="bottom")
ax2.set_ylim(0, 13_000)
ax2.set_ylabel("Units sold")
ax2.set_title("Units sold, February", loc="left", fontsize=11)
ax2.set_xticks(years)
for a in (ax1, ax2):
    a.grid(axis="y", alpha=0.25)
fig.suptitle("Greater Toronto Area housing market, February 2020-2023",
             weight="bold", x=0.02, ha="left")
fig.text(0.02, 0.005, "Source: TRREB (prices approximate, read from the original chart)",
         fontsize=8, color="grey")
fig.tight_layout()
fig.savefig("charts/q1c_redesign_sketch.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 6.2  "TECH STOCK THROWBACK?"  NETFLIX (2011-13) vs TESLA (2019-TODAY)
# =============================================================================
"""
6.2 (a) Two major issues

ISSUE 1 - Dual y-axes with unrelated, non-zero scales.
  Netflix uses the left axis (0-50) and Tesla the right axis (160-360, which
  does not start at zero). The designer can slide/stretch either axis freely,
  so the two curves can be made to "overlap" arbitrarily. The visual similarity
  is manufactured by the axis choice, not by the data (graphical integrity).
  Readers also have to work out which line belongs to which axis (only colour
  links them), and the absolute price levels are not comparable.

ISSUE 2 - Two different time periods forced onto one unlabelled x-axis.
  Netflix covers Jun 2011 - Nov 2013 and Tesla Jan 2019 - "today"; the x-axis
  has no dates or tick labels, and the two series do not even have the same
  length (Tesla's line stops early).  The chart implies that the time spans and
  speeds are comparable and invites a spurious prediction ("Tesla will rebound
  like Netflix").
  (Minor: chartjunk - glowing lines, dark gradient background, title in
   question form that suggests a conclusion the data do not support.)

6.2 (b) Alternative design - indexed performance on a common relative-time axis
  * Normalise each stock to its own starting price:  index = price / price_0 x 100
    (or % change from start). Both series then share ONE y-axis with the same
    meaning ("% of starting price"), removing the dual-axis manipulation.
  * Put both on a common x-axis "trading days (or months) since start of
    period", clearly labelled, and state each period in a direct label at the
    end of each line (no legend, no colour-to-axis matching needed).
  * Optionally show the absolute prices in two small multiples underneath, each
    with its own real date axis, so the actual calendar periods are visible.
  * A reference line at 100 (= starting value) makes gains/losses obvious.
  Advantages: honest comparison of relative movement (what the "throwback"
  story is really about), explicit time, one scale to read, and the analogy is
  shown as an analogy rather than as a forecast.

  -> charts/q2b_redesign_indexed.png  (schematic with SYNTHETIC series that only
     mimic the shape of the original chart - not real stock prices)
"""

rng = np.random.default_rng(6)


def synthetic_path(n, start, drift_profile, vol):
    """Smooth random walk following a drift profile (illustration only)."""
    steps = drift_profile + rng.normal(0, vol, n)
    return start * np.exp(np.cumsum(steps))


n_nf, n_ts = 620, 120   # ~2.5 years vs ~6 months of trading days
t = np.linspace(0, 1, n_nf)
nf_profile = np.where(t < 0.25, -0.0045, np.where(t < 0.55, 0.0002, 0.0042))
netflix = synthetic_path(n_nf, 42, nf_profile, 0.018)
tesla = synthetic_path(n_ts, 340, np.full(n_ts, -0.0055), 0.022)

fig, (ax_top, ax_l, ax_r) = plt.subplots(
    3, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [1.5, 1, 1]})
for s, name, col in [(netflix, "Netflix  (Jun 2011 - Nov 2013)", "#d6604d"),
                     (tesla, "Tesla  (Jan 2019 - ...)", "#4393c3")]:
    idx = s / s[0] * 100
    ax_top.plot(np.arange(len(s)), idx, color=col, lw=1.8)
    ax_top.annotate(name, (len(s) - 1, idx[-1]), xytext=(8, -14 if len(s) < n_nf else 0),
                    textcoords="offset points", color=col, va="center", fontsize=9,
                    bbox=dict(fc="white", ec="none", alpha=0.8))
ax_top.axhline(100, color="grey", lw=0.8, ls="--")
ax_top.text(n_nf + 160, 102, "start = 100", fontsize=8, color="grey", ha="right")
ax_top.set_xlim(0, n_nf + 170)
ax_top.set_ylim(30, None)
ax_top.set_ylabel("Price index (start = 100)")
ax_top.set_xlabel("Trading days since start of period")
ax_top.set_title("Main view: one shared y-axis (indexed), one shared relative-time axis",
                 loc="left", fontsize=11)

ax_l.plot(np.arange(n_nf), netflix, color="#d6604d")
ax_l.set_title("Netflix - absolute price, real dates (2011-2013)", loc="left", fontsize=10)
ax_l.set_ylim(0, None)
ax_l.set_xticks(np.linspace(0, n_nf - 1, 5), ["Jun 2011", "Dec 2011", "Jun 2012", "Dec 2012", "Nov 2013"])
ax_r.plot(np.arange(n_ts), tesla, color="#4393c3")
ax_r.set_title("Tesla - absolute price, real dates (2019-)", loc="left", fontsize=10)
ax_r.set_ylim(0, None)
ax_r.set_xticks(np.linspace(0, n_ts - 1, 4), ["Jan 2019", "Mar 2019", "May 2019", "Jun 2019"])
ax_r.set_xlim(ax_l.get_xlim()[0] * n_ts / n_nf, (ax_l.get_xlim()[1]) * n_ts / n_nf)
for a in (ax_l, ax_r):
    a.set_ylabel("USD")
fig.suptitle("6.2(b) Redesign schematic - SYNTHETIC data, layout illustration only",
             weight="bold", color="#444444")
fig.tight_layout()
fig.savefig("charts/q2b_redesign_indexed.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 6.3  INTERACTION IN COMPENDIA  (query: "TikTok trends worldwide")
# =============================================================================
"""
Taxonomy used in the lecture: Yi, Kang, Stasko & Jacko (2007)
  Select / Explore / Reconfigure / Encode / Abstract-Elaborate / Filter / Connect

6.3 (a) Interaction types used in Compendia
| Type               | Where it appears in Compendia                                        |
|--------------------|----------------------------------------------------------------------|
| Select             | Clicking a Thematic Circle (e.g. "Youth Appeal") marks that theme as |
|                    | the focus and jumps into its story; hovering a fact dot marks it.    |
| Explore            | Scrolling (scrollytelling) moves the viewport to the next narrative  |
|                    | unit; animated pan & zoom from the overview to a cluster.            |
| Abstract/Elaborate | Overview -> Story View: Thematic Overview (one representative fact   |
|                    | per theme) vs. zoomed circle with top-3 facts + narrative + chart;   |
|                    | hover a dot -> detailed fact panel (details-on-demand); click outside|
|                    | the circle to go back to the abstract overview.                      |
| Filter             | Filter widget: relevance slider (number of themes shown, 6 default), |
|                    | shared-article range slider (hide weak links), toggle fact dots.     |
| Connect            | Links between circles show shared source articles; hovering a link   |
|                    | lists the shared articles; facts in the caption are highlighted with |
|                    | the same colour as the matching dots; similar facts are connected.   |
| Reconfigure        | Limited - the layout of the overview changes when filters change the |
|                    | number of visible themes; story order is fixed by relevance.         |
| Encode             | Not really supported - the user cannot change chart types; the       |
|                    | system chooses bar/pie/line/isotype/dumbbell automatically.          |
Also: clicking a favicon opens the original article (source traceability).

6.3 (b) Most useful type: ABSTRACT/ELABORATE (overview -> details on demand),
  closely followed by Explore through scrolling.
  Justification with examples:
  * For "TikTok trends worldwide" Compendia extracts dozens of facts from many
    articles. Showing them all at once would be overwhelming. The overview shows
    only ~6 themes (Global Reach, Gen Z Focus, App Usage, Youth Appeal ...),
    each with one representative fact and radial bars for #facts/#articles, so
    the user first gets the big picture (Shneiderman's mantra).
  * Elaborating on "Gen Z Focus" reveals the narrative "~70% of TikTok's users
    are Gen Z" with a chart and the top-3 supporting facts; hovering a fact dot
    elaborates further to the exact sentence, source and year. This is how the
    user verifies credibility - the main analytical need when facts come from
    heterogeneous web articles.
  * Abstracting back (click outside the circle) returns to the overview to pick
    another theme, so the user can alternate between context and detail without
    losing orientation.
  (Filter is useful but secondary: the default of 6 themes is usually adequate.)

6.3 (c) Animation: YES.
  * Animated pan-and-zoom transition from the Thematic Overview into a Story
    View when scrolling starts or a circle is clicked.
  * Within a cluster, the zoomed Thematic Circle animates to re-highlight the
    facts that support the current narrative unit as the user scrolls.
  Helpful? Mostly yes:
  + Object constancy / context preservation: the user sees WHERE the story
    being read sits inside the overview (the zoom starts from the circle),
    reducing disorientation when switching between abstraction levels
    (congruent with the semantic change "zoom into this theme").
  + Smooth highlight transitions make it easy to notice which facts changed
    between consecutive narrative units (change blindness is avoided).
  + Scroll-driven animation is user-paced, so it respects the apprehension
    principle better than auto-playing animation.
  - Caveats: transitions that are too long slow down expert users who want to
    jump between themes; fast scrolling can trigger many transitions and make
    the view feel jumpy; animation adds little for precise comparison of values
    (static charts in the narrative units do that job).
"""

# =============================================================================
# 6.4  CONGRUENCE vs. APPREHENSION
# =============================================================================
"""
Congruence  : visual change must correspond to a real change in the data/meaning.
Apprehension: the animation must be easy to perceive and understand.

| Case | Principle    | Followed / Violated | Reason                                                      |
|------|--------------|---------------------|-------------------------------------------------------------|
| (a)  | Apprehension | FOLLOWED            | Group similar transitions + staging reduce what the eye     |
|      |              |                     | must track at once.                                         |
| (b)  | Apprehension | FOLLOWED            | Minimising occlusion keeps every mark trackable.            |
| (c)  | Apprehension | VIOLATED            | Complex, unpredictable curved paths break "maximise         |
|      |              |                     | predictability / use simple transitions" (straight lines).  |
| (d)  | Congruence   | FOLLOWED            | Bar height interpolates 50 -> 80: visual change mirrors the |
|      |              |                     | data change; graphic stays valid during the transition.     |
| (e)  | Congruence   | FOLLOWED            | Object constancy / semantic correspondence - the same bar   |
|      |              |                     | keeps representing the same product while it moves.         |
| (f)  | Congruence   | VIOLATED            | Morphing points into bars with no shared data dimension     |
|      |              |                     | implies a correspondence that does not exist (ambiguity).   |
| (g)  | Congruence   | VIOLATED            | Same syntax (height-change animation) used for a different  |
|      |              |                     | semantic (change of measure, not of time) -> viewers think  |
|      |              |                     | sales changed. Use a distinct transition, e.g. relabel axis |
|      |              |                     | first, fade/cross-dissolve, or staged axis rescale.         |
"""

print("Tutorial 6 charts saved to ./charts/")

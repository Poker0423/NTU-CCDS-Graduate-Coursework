"""
SD6105 Data Visualisation - Tutorial 7 Solutions
Topic: High-Dimensional Data Visualization

7.1  1,000 e-commerce customers, 6 numerical + 3 categorical attributes
     (a) multivariate pattern / cluster analysis
     (b) relationships between categorical variables
     (c) comparison of individual customer profiles

The handout gives no data file, so a SYNTHETIC dataset with the requested
attributes is generated (fixed seed, 4 hidden customer segments) and saved to
./data/customers_synthetic.csv. All charts go to ./charts/.

Run from inside this folder:  python tutorial7_solutions.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

os.makedirs("charts", exist_ok=True)
os.makedirs("data", exist_ok=True)
sns.set_theme(style="white", context="notebook")

NUM = ["Age", "Income", "Spending Score", "Purchases", "Time on Site", "Satisfaction"]
CAT = ["Gender", "Category", "Region"]

# =============================================================================
# 0. SYNTHETIC DATA (4 latent segments)
# =============================================================================
rng = np.random.default_rng(7)
segments = {
    #                  Age  Income(k) Spend Purch  Time(min) Satisf  weight
    "Young bargain":  (24,   35,      70,   28,    38,       3.6,   0.30),
    "Affluent pro":   (41,  115,      78,   22,    18,       4.3,   0.20),
    "Family saver":   (38,   62,      35,   12,    22,       3.4,   0.30),
    "Senior casual":  (63,   55,      25,    6,    12,       4.0,   0.20),
}
sd = np.array([4, 12, 9, 4, 7, 0.45])
cat_probs = {   # P(category) per segment  ->  Electronics, Fashion, Home, Grocery, Books
    "Young bargain": [0.30, 0.40, 0.05, 0.10, 0.15],
    "Affluent pro":  [0.40, 0.25, 0.20, 0.05, 0.10],
    "Family saver":  [0.10, 0.15, 0.35, 0.35, 0.05],
    "Senior casual": [0.05, 0.05, 0.25, 0.30, 0.35],
}
region_probs = {  # North, South, East, West
    "Young bargain": [0.20, 0.20, 0.40, 0.20],
    "Affluent pro":  [0.15, 0.15, 0.30, 0.40],
    "Family saver":  [0.30, 0.35, 0.15, 0.20],
    "Senior casual": [0.35, 0.30, 0.15, 0.20],
}
female_share = {"Young bargain": 0.60, "Affluent pro": 0.45,
                "Family saver": 0.55, "Senior casual": 0.50}
cats = ["Electronics", "Fashion", "Home", "Grocery", "Books"]
regions = ["North", "South", "East", "West"]

rows, n_total = [], 1000
for name, (*mu, w) in segments.items():
    n = int(round(w * n_total))
    X = rng.normal(mu, sd, size=(n, 6))
    for x in X:
        rows.append({
            "Age": int(np.clip(x[0], 18, 80)),
            "Income": round(float(np.clip(x[1], 12, 200)), 1),          # k$/year
            "Spending Score": int(np.clip(x[2], 1, 100)),
            "Purchases": int(np.clip(x[3], 0, 60)),
            "Time on Site": round(float(np.clip(x[4], 1, 90)), 1),       # min/visit
            "Satisfaction": round(float(np.clip(x[5], 1, 5)), 1),        # 1-5
            "Gender": "Female" if rng.random() < female_share[name] else "Male",
            "Category": rng.choice(cats, p=cat_probs[name]),
            "Region": rng.choice(regions, p=region_probs[name]),
            "_true_segment": name,
        })
df = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
df.insert(0, "CustomerID", [f"C{i:04d}" for i in range(1, len(df) + 1)])
df.drop(columns="_true_segment").to_csv("data/customers_synthetic.csv", index=False)
print(f"Dataset: {df.shape[0]} customers x {len(NUM) + len(CAT)} attributes")

# =============================================================================
# 7.1 (a)  MULTIVARIATE PATTERN ANALYSIS
# =============================================================================
"""
Design: PARALLEL COORDINATES (main view) + SCATTER-PLOT MATRIX + PCA projection,
coordinated by a colour = cluster label from k-means on standardised features.

Why parallel coordinates?
 * Shows ALL 6 numerical dimensions at once for all 1,000 customers; each
   customer = one polyline. Clusters appear as bundles of lines that run
   through similar ranges on every axis; negative correlation between adjacent
   axes appears as crossing lines (e.g. Age vs Time on Site), positive as
   parallel lines.
 * Scales well in #dimensions; supports brushing on an axis (interactive) to
   isolate a segment.
 Limitations & mitigations: overplotting with 1,000 lines -> low alpha + draw
 cluster medians as thick lines; pattern depends on axis ORDER -> order axes
 so that related attributes are adjacent; axes normalised to [0, 1].
Why add a scatter-plot matrix?  It shows every PAIRWISE relationship and its
 strength precisely (position on common scales), complementing PCP.
Why PCA?  A 2-D overview where distance ~ similarity across all 6 attributes;
 well-separated groups confirm the clusters.
"""
X = StandardScaler().fit_transform(df[NUM])
km = KMeans(n_clusters=4, n_init=20, random_state=7).fit(X)
# give clusters readable names, ordered by mean age
order = df.assign(k=km.labels_).groupby("k")["Age"].mean().sort_values().index
names = {k: f"Cluster {i+1}" for i, k in enumerate(order)}
df["Cluster"] = pd.Categorical([names[k] for k in km.labels_],
                               categories=[f"Cluster {i}" for i in range(1, 5)])
PAL = dict(zip(df["Cluster"].cat.categories, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]))
print(df.groupby("Cluster", observed=True)[NUM].mean().round(1))

# --- Parallel coordinates ------------------------------------------------------
axis_order = ["Age", "Time on Site", "Purchases", "Spending Score", "Income", "Satisfaction"]
norm = pd.DataFrame(MinMaxScaler().fit_transform(df[axis_order]), columns=axis_order)
fig, ax = plt.subplots(figsize=(12, 5.5))
xs = np.arange(len(axis_order))
for cl, col in PAL.items():
    sub = norm[df["Cluster"] == cl]
    ax.plot(xs, sub.T.values, color=col, alpha=0.06, lw=0.8)
    ax.plot(xs, sub.median().values, color=col, lw=3.5, label=f"{cl} (median)")
for x, a in zip(xs, axis_order):
    ax.axvline(x, color="black", lw=0.8)
    lo, hi = df[a].min(), df[a].max()
    ax.text(x, -0.06, f"{lo:g}", ha="center", va="top", fontsize=8)
    ax.text(x, 1.03, f"{hi:g}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs, axis_order, fontsize=11)
ax.xaxis.tick_top()
ax.tick_params(axis="x", pad=18)
ax.set_yticks([])
ax.set_ylim(-0.12, 1.12)
sns.despine(left=True, bottom=True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
ax.set_title("7.1(a) Parallel coordinates of 1,000 customers, coloured by k-means cluster",
             pad=40, weight="bold")
fig.tight_layout()
fig.savefig("charts/q1a_parallel_coordinates.png", bbox_inches="tight")
plt.close(fig)

# --- Scatter-plot matrix -------------------------------------------------------
g = sns.pairplot(df, vars=NUM, hue="Cluster", palette=PAL, corner=True,
                 plot_kws=dict(s=8, alpha=0.5, linewidth=0), diag_kind="kde", height=1.9)
g.fig.suptitle("7.1(a) Scatter-plot matrix (pairwise relationships)", y=1.01, weight="bold")
g.savefig("charts/q1a_scatterplot_matrix.png", bbox_inches="tight")
plt.close(g.fig)

# --- PCA projection ------------------------------------------------------------
pca = PCA(n_components=2).fit(X)
P = pca.transform(X)
fig, ax = plt.subplots(figsize=(7.5, 6))
for cl, col in PAL.items():
    m = (df["Cluster"] == cl).values
    ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.6, color=col, label=cl)
for i, f in enumerate(NUM):  # loadings as arrows (biplot)
    vx, vy = pca.components_[0, i] * 3, pca.components_[1, i] * 3
    ax.arrow(0, 0, vx, vy, color="black", width=0.01, head_width=0.08, alpha=0.7)
    ax.text(vx * 1.12, vy * 1.12, f, fontsize=9, ha="center")
ev = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f"PC1 ({ev[0]:.0f}% variance)")
ax.set_ylabel(f"PC2 ({ev[1]:.0f}% variance)")
ax.legend(frameon=False, markerscale=2)
ax.set_title("7.1(a) PCA biplot - 6-D customers projected to 2-D", weight="bold")
sns.despine()
fig.tight_layout()
fig.savefig("charts/q1a_pca_biplot.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 7.1 (b)  CATEGORICAL RELATIONSHIP ANALYSIS
# =============================================================================
"""
Design: PARALLEL SETS (Kosara et al.) for Gender -> Region -> Category,
plus a small-multiple HEATMAP of Region x Category (row %) for each gender.

Why parallel sets?
 * Designed for multivariate CATEGORICAL data: each axis is a variable, each
   box a category whose width = frequency; ribbons between axes show how many
   customers share each combination, so joint and conditional distributions
   (e.g. "of West customers, how many prefer Electronics?") are visible.
 * Much less cluttered than parallel coordinates for categorical data because
   identical records are aggregated into one ribbon.
 Limitations: ribbon crossings grow with #categories; only adjacent axes are
 directly comparable -> allow axis re-ordering (interaction) / colour ribbons
 by the variable of interest.
Why the heatmap?  Exact proportions of a 2-way contingency table (read like a
 table but pre-attentive via colour); faceting by gender adds the 3rd variable.
 (Mosaic plots are an equally valid alternative.)
"""


def parallel_sets(ax, data, dims, colour_dim, palette, gap=0.02):
    """Minimal parallel-sets implementation with matplotlib."""
    n = len(data)
    pos = {}
    for j, d in enumerate(dims):
        counts = data[d].value_counts().sort_index()
        y = 0
        for cat, c in counts.items():
            h = c / n
            pos[(d, cat)] = [y, y + h]
            ax.add_patch(plt.Rectangle((j - 0.04, y), 0.08, h, color="#444444"))
            ax.text(j + (0.07 if j < len(dims) - 1 else -0.07), y + h / 2,
                    f"{cat} ({c})", va="center",
                    ha="left" if j < len(dims) - 1 else "right", fontsize=9)
            y += h + gap
        ax.text(j, -0.06, d, ha="center", weight="bold", fontsize=11)
    for j in range(len(dims) - 1):
        a, b = dims[j], dims[j + 1]
        keys = list(dict.fromkeys([colour_dim, a, b]))   # colour_dim may equal a or b
        grp = data.groupby(keys, observed=True).size().reset_index(name="n")
        grp = grp.sort_values(list(dict.fromkeys([a, colour_dim, b])))
        off_a = {k: v[0] for k, v in pos.items() if k[0] == a}
        off_b = {k: v[0] for k, v in pos.items() if k[0] == b}
        # sort ribbons on the right side by (b, colour, a) for tidiness
        grp_b = grp.sort_values(list(dict.fromkeys([b, colour_dim, a])))
        right_y = {}
        for _, r in grp_b.iterrows():
            key = (r[colour_dim], r[a], r[b])
            right_y[key] = off_b[(b, r[b])]
            off_b[(b, r[b])] += r["n"] / n
        for _, r in grp.iterrows():
            h = r["n"] / n
            y0 = off_a[(a, r[a])]
            off_a[(a, r[a])] += h
            y1 = right_y[(r[colour_dim], r[a], r[b])]
            x0, x1 = j + 0.04, j + 1 - 0.04
            xm = (x0 + x1) / 2
            verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1),
                     (x1, y1 + h), (xm, y1 + h), (xm, y0 + h), (x0, y0 + h), (x0, y0)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), fc=palette[r[colour_dim]],
                                   ec="none", alpha=0.55))
    ax.set_xlim(-0.3, len(dims) - 0.7)
    ax.set_ylim(1.12, -0.1)
    ax.axis("off")


cat_pal = dict(zip(cats, sns.color_palette("Set2", len(cats)).as_hex()))
fig, ax = plt.subplots(figsize=(12, 6.5))
parallel_sets(ax, df, ["Gender", "Region", "Category"], "Category", cat_pal)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in cat_pal.values()]
ax.legend(handles, cat_pal.keys(), title="Ribbon colour = Category", frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5)
ax.set_title("7.1(b) Parallel sets: Gender -> Region -> Product category preference",
             weight="bold")
fig.tight_layout()
fig.savefig("charts/q1b_parallel_sets.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, gdr in zip(axes, ["Female", "Male"]):
    ct = pd.crosstab(df.loc[df.Gender == gdr, "Region"], df.loc[df.Gender == gdr, "Category"],
                     normalize="index").reindex(index=regions, columns=cats) * 100
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=45, cbar=ax is axes[1],
                cbar_kws={"label": "% of customers in region"}, ax=ax, linewidths=0.5)
    ax.set_title(f"{gdr} (n={int((df.Gender == gdr).sum())})")
    ax.set_xlabel("Preferred category")
fig.suptitle("7.1(b) Region x Category (row %), faceted by Gender", weight="bold")
fig.tight_layout()
fig.savefig("charts/q1b_region_category_heatmaps.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 7.1 (c)  INDIVIDUAL PROFILE COMPARISON
# =============================================================================
"""
Design: RADAR (STAR) CHART small multiples - one glyph per customer - with the
population median drawn as a grey reference polygon, plus an overlay view of
the selected customers for direct comparison.

Why radar/star glyphs?
 * A glyph-based method: every customer becomes ONE compact shape, so analysts
   compare whole profiles at a glance (shape similarity = profile similarity).
 * The median polygon gives context ("above/below typical on which attribute").
 * Small multiples avoid the clutter of overlaying many polygons; the overlay
   panel supports precise pairwise comparison of a few selected customers.
 Notes: all axes are min-max normalised to [0, 1] so different units are
 comparable; area exaggerates differences and axis order changes the shape, so
 keep a fixed order and few (<=4) customers per overlay. Categorical attributes
 (gender, category, region) are shown as text in each panel title.
 Alternatives: parallel coordinates with the selected customers highlighted,
 or a heatmap table (rows = customers, columns = attributes).
"""
sel_ids = []
for cl in df["Cluster"].cat.categories:     # one representative per cluster ...
    sub = df[df.Cluster == cl]
    sel_ids.append(sub.index[0])
sel_ids += [df["Income"].idxmax(), df["Satisfaction"].idxmin()]  # ... + 2 extremes
radar_axes = axis_order
norm_all = pd.DataFrame(MinMaxScaler().fit_transform(df[radar_axes]), columns=radar_axes)
angles = np.linspace(0, 2 * np.pi, len(radar_axes), endpoint=False).tolist()
angles_c = angles + angles[:1]
median = norm_all.median().tolist()

fig, axes = plt.subplots(2, 3, figsize=(13, 9), subplot_kw=dict(polar=True))
for ax, i in zip(axes.flat, sel_ids):
    vals = norm_all.loc[i].tolist()
    col = PAL[df.loc[i, "Cluster"]]
    ax.fill(angles_c, median + median[:1], color="grey", alpha=0.18)
    ax.plot(angles_c, median + median[:1], color="grey", lw=1, ls="--")
    ax.fill(angles_c, vals + vals[:1], color=col, alpha=0.35)
    ax.plot(angles_c, vals + vals[:1], color=col, lw=2)
    ax.set_xticks(angles, radar_axes, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75], [], )
    ax.set_ylim(0, 1)
    r = df.loc[i]
    ax.set_title(f"{r.CustomerID} - {r.Cluster}\n{r.Gender}, {r.Region}, likes {r.Category}\n"
                 f"Age {r.Age}, Income ${r.Income:.0f}k", fontsize=9, pad=16)
fig.suptitle("7.1(c) Customer profile glyphs (radar charts); grey dashed = population median",
             weight="bold")
fig.tight_layout()
fig.savefig("charts/q1c_radar_profiles.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i in sel_ids[:4]:
    vals = norm_all.loc[i].tolist()
    ax.plot(angles_c, vals + vals[:1], lw=2, color=PAL[df.loc[i, "Cluster"]],
            label=f"{df.loc[i, 'CustomerID']} ({df.loc[i, 'Cluster']})")
    ax.fill(angles_c, vals + vals[:1], color=PAL[df.loc[i, "Cluster"]], alpha=0.08)
ax.set_xticks(angles, radar_axes)
ax.set_yticklabels([])
ax.set_ylim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
ax.set_title("7.1(c) Overlay of 4 selected customers", weight="bold", pad=20)
fig.tight_layout()
fig.savefig("charts/q1c_radar_overlay.png", bbox_inches="tight")
plt.close(fig)

print("Tutorial 7 charts saved to ./charts/")

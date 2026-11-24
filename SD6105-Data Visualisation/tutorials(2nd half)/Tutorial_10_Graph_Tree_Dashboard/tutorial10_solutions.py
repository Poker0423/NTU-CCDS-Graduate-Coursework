"""
SD6105 Data Visualisation - Tutorial 10 Solutions
Topic: Graph Visualization, Tree (Hierarchical) Visualization, Dashboard Design

10.1  Dense graph with 100 nodes: node-link diagram vs adjacency matrix
10.2  University hierarchy: node-link vs sunburst vs treemap
10.3  Problems of the bicycle-manufacturing dashboard

Supporting charts are generated with SYNTHETIC data (fixed seed) in ./charts/.
Run from inside this folder:  python tutorial10_solutions.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

os.makedirs("charts", exist_ok=True)
os.makedirs("data", exist_ok=True)
rng = np.random.default_rng(10)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 110})

# =============================================================================
# 10.1  DENSE GRAPH: NODE-LINK vs ADJACENCY MATRIX
# =============================================================================
"""
Choice: ADJACENCY MATRIX.

Justification
 * A 100-node dense graph can have thousands of edges (density 0.3 -> ~1,500
   edges). In a node-link diagram the edges overlap and cross everywhere
   ("hairball"): edge crossings, node-edge occlusion, and no way to tell
   whether two particular nodes are connected. Force-directed layouts do not
   untangle dense graphs.
 * In a matrix every possible edge has its OWN cell (100 x 100), so there are
   no crossings or occlusion regardless of density; the visual complexity is
   constant (n^2 cells) and dense graphs actually USE the space well.
 * Tasks that become easy: "are A and B connected?" (look up one cell),
   finding clusters/cliques (dense blocks on the diagonal after re-ordering),
   comparing degrees (row darkness), encoding edge weight (cell colour).
 * Node labels are readable along the rows/columns without overlap.

Is it perfect? No - limitations:
 * Strongly depends on ROW/COLUMN ORDERING: with a random order the clusters
   are invisible -> need seriation / community-based or clustering-based
   reordering (illustrated below).
 * PATH-related tasks (e.g. shortest path between two nodes, following a
   chain of connections) are much harder than in node-link diagrams because
   the eye must jump between rows and columns.
 * Less intuitive for lay users; uses n^2 space even for sparse parts, so
   very large graphs (10k nodes) need aggregation; labels need space too.
 * Topological "shape" (bridges, periphery) is not visible.
 Mitigations: hybrid views (e.g. NodeTrix: matrices for dense communities +
 node-link for inter-community links), interactive highlighting of rows /
 columns, and path overlay on hover.
"""
sizes = [30, 25, 25, 20]
p_in, p_out = 0.55, 0.12
G = nx.stochastic_block_model(sizes, [[p_in if i == j else p_out for j in range(4)]
                                      for i in range(4)], seed=10)
community = np.concatenate([[k] * s for k, s in enumerate(sizes)])
perm = rng.permutation(100)                       # hide the structure
G = nx.relabel_nodes(G, {old: int(new) for old, new in zip(range(100), perm)})
comm_of = {int(perm[i]): community[i] for i in range(100)}
print(f"10.1  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"density {nx.density(G):.2f}")

COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
pos = nx.spring_layout(G, seed=10)
nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.15, width=0.5)
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=30, node_color="#555555")
axes[0].set_title("Node-link (force-directed): a 'hairball'\n"
                  "edges overlap; is node A linked to B?", fontsize=11)
axes[0].axis("off")

A = nx.to_numpy_array(G, nodelist=range(100))
axes[1].imshow(A, cmap="Greys", interpolation="nearest")
axes[1].set_title("Adjacency matrix - arbitrary node order\nno crossings, but no visible structure",
                  fontsize=11)
order = sorted(range(100), key=lambda n: (comm_of[n], -G.degree(n)))
A2 = nx.to_numpy_array(G, nodelist=order)
axes[2].imshow(A2, cmap="Greys", interpolation="nearest")
start = 0
for k, s in enumerate(sizes):
    axes[2].add_patch(plt.Rectangle((start - 0.5, start - 0.5), s, s, fill=False,
                                    ec=COLS[k], lw=2))
    start += s
axes[2].set_title("Adjacency matrix - reordered by community\n4 dense blocks = 4 clusters",
                  fontsize=11)
for ax in axes[1:]:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("10.1  Dense graph (100 nodes): node-link vs adjacency matrix", weight="bold")
fig.tight_layout()
fig.savefig("charts/q1_nodelink_vs_matrix.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 10.2  UNIVERSITY HIERARCHY
# =============================================================================
"""
(a) Task -> technique
| Task                                              | Choice            | Why                                                      |
|---------------------------------------------------|-------------------|----------------------------------------------------------|
| Understand hierarchical relationships between     | NODE-LINK diagram | Parent-child links are drawn EXPLICITLY, one node per    |
| colleges, schools, departments, faculty members   | (tidy tree)       | unit, depth = level; easy to follow a path university -> |
|                                                   |                   | college -> school -> dept -> person, read labels, see    |
|                                                   |                   | fan-out. Structure is the focus, not quantities.         |
| Compare total research funding across departments | TREEMAP           | Space-filling: rectangle AREA = funding, nested in the   |
| and colleges                                      |                   | parent; colleges' totals and departments' shares are     |
|                                                   |                   | both visible, colour can encode a 2nd attribute (e.g.    |
|                                                   |                   | #faculty). Rectangles are easier to compare than arcs.   |
| Overview of how units are distributed across      | SUNBURST          | Rings = hierarchy levels (university at centre, then     |
| hierarchy levels                                  |                   | colleges, schools, departments outward), so the number   |
|                                                   |                   | of units at each LEVEL and how each parent splits is     |
|                                                   |                   | seen at a glance; levels are explicit (treemaps hide     |
|                                                   |                   | internal levels), compact radial overview.               |

(b) Treemap vs node-link for a LARGE hierarchy
 Advantages of treemap
  + Space-filling: uses 100% of the screen, scales to thousands of leaves;
    node-link wastes space (white space between nodes) and runs out of
    horizontal room at the deep / wide levels (leaves overlap).
  + Encodes a quantitative attribute (size by area, plus colour) for every
    node - ideal for "which parts are biggest?" questions.
  + Part-whole relationships are immediate (children tile their parent).
 Disadvantages of treemap
  - Hierarchical STRUCTURE is implicit (nesting only): hard to tell depth,
    to see internal nodes, or to trace parent-child paths.
  - Area/aspect-ratio comparisons are less accurate than position/length;
    thin slivers (slice-and-dice) or unstable layouts (squarified) hurt
    comparison; small nodes become invisible and cannot be labelled.
  - Needs a size attribute; units with zero value disappear.
 Node-link: explicit structure & paths, but poor scalability and no natural
 encoding of size.
"""
colleges = {
    "Engineering": {"Computing": ["AI", "Software Eng.", "Data Science", "Cyber Security"],
                    "Mech & Aero": ["Mechanical", "Aerospace"],
                    "EEE": ["Power Systems", "Microelectronics", "Communications"]},
    "Science": {"Physical Sci.": ["Physics", "Mathematics"],
                "Biological Sci.": ["Biology", "Biochemistry", "Ecology"],
                "Chemistry": ["Organic Chem.", "Materials"]},
    "Business": {"Business School": ["Finance", "Marketing", "Strategy", "Accounting"]},
    "Humanities": {"Social Sci.": ["Economics", "Psychology", "Sociology"],
                   "Arts": ["History", "Linguistics"]},
}
rows = []
for c, schools in colleges.items():
    for s, depts in schools.items():
        for d in depts:
            fac = int(rng.integers(12, 60))
            fund = round(fac * rng.uniform(0.15, 0.9) * (2.0 if c == "Engineering" else 1.0), 1)
            rows.append((c, s, d, fac, fund))
uni = pd.DataFrame(rows, columns=["College", "School", "Department", "Faculty", "Funding_SGDm"])
uni.to_csv("data/university_synthetic.csv", index=False)
CC = dict(zip(colleges, ["#4e79a7", "#59a14f", "#f28e2b", "#b07aa1"]))

# ---- node-link tree (tidy, top-down) ----------------------------------------
fig, ax = plt.subplots(figsize=(16, 6.5))
leaves = uni.reset_index(drop=True)
x_leaf = {r.Department: i for i, r in leaves.iterrows()}
x_school = leaves.groupby("School", sort=False).apply(lambda g: g.index.to_series().mean())
x_col = leaves.groupby("College", sort=False).apply(lambda g: g.index.to_series().mean())
x_root = np.mean(list(x_col.values))
Y = {"uni": 3, "col": 2, "sch": 1, "dep": 0}
for c, xc in x_col.items():
    ax.plot([x_root, xc], [Y["uni"], Y["col"]], color="grey", lw=1)
for (s, xs), c in zip(x_school.items(), leaves.groupby("School", sort=False)["College"].first()):
    ax.plot([x_col[c], xs], [Y["col"], Y["sch"]], color="grey", lw=1)
for _, r in leaves.iterrows():
    ax.plot([x_school[r.School], x_leaf[r.Department]], [Y["sch"], Y["dep"]], color="grey", lw=1)
    ax.scatter(x_leaf[r.Department], Y["dep"], s=r.Faculty * 4, color=CC[r.College], zorder=3)
    ax.text(x_leaf[r.Department], Y["dep"] - 0.15, r.Department, rotation=90, ha="center",
            va="top", fontsize=8)
for s, xs in x_school.items():
    c = leaves.loc[leaves.School == s, "College"].iloc[0]
    ax.scatter(xs, Y["sch"], s=120, color=CC[c], zorder=3)
    ax.text(xs, Y["sch"] + 0.12, s, ha="center", fontsize=8)
for c, xc in x_col.items():
    ax.scatter(xc, Y["col"], s=220, color=CC[c], zorder=3)
    ax.text(xc, Y["col"] + 0.14, c, ha="center", fontsize=10, weight="bold")
ax.scatter(x_root, Y["uni"], s=320, color="#333333", zorder=3)
ax.text(x_root, Y["uni"] + 0.15, "University", ha="center", fontsize=11, weight="bold")
for lvl, y in [("University", 3), ("College", 2), ("School", 1), ("Department", 0)]:
    ax.text(-1.8, y, lvl, ha="right", va="center", fontsize=9, color="grey", style="italic")
ax.set_xlim(-4, len(leaves))
ax.set_ylim(-1.6, 3.5)
ax.axis("off")
ax.set_title("10.2 Node-link tree - hierarchical relationships (leaf size = #faculty; "
             "faculty members would be a 5th level)", weight="bold", loc="left")
fig.tight_layout()
fig.savefig("charts/q2_nodelink_tree.png", bbox_inches="tight")
plt.close(fig)


# ---- squarified treemap (no external package) --------------------------------
def squarify(values, x, y, w, h):
    """Bruls et al. squarified treemap. values must be sorted descending."""
    values = list(values)
    total = sum(values)
    vals = [v * w * h / total for v in values]
    rects = []

    def worst(row, side):
        s = sum(row)
        return max(max(side ** 2 * r / s ** 2, s ** 2 / (side ** 2 * r)) for r in row)

    def layout(row, x, y, w, h):
        s = sum(row)
        out = []
        if w >= h:                       # place a column on the left
            cw = s / h
            yy = y
            for r in row:
                out.append((x, yy, cw, r / cw))
                yy += r / cw
            return out, (x + cw, y, w - cw, h)
        rh = s / w                       # place a row at the top
        xx = x
        for r in row:
            out.append((xx, y, r / rh, rh))
            xx += r / rh
        return out, (x, y + rh, w, h - rh)

    row = []
    while vals:
        side = min(w, h)
        if not row or worst(row + [vals[0]], side) <= worst(row, side):
            row.append(vals.pop(0))
        else:
            out, (x, y, w, h) = layout(row, x, y, w, h)
            rects += out
            row = []
    if row:
        out, _ = layout(row, x, y, w, h)
        rects += out
    return rects


fig, ax = plt.subplots(figsize=(13, 8))
col_tot = uni.groupby("College")["Funding_SGDm"].sum().sort_values(ascending=False)
for (c, tot), (cx, cy, cw, ch) in zip(col_tot.items(), squarify(col_tot.values, 0, 0, 100, 60)):
    sub = uni[uni.College == c].sort_values("Funding_SGDm", ascending=False)
    pad = 0.6
    for (_, r), (x, y, w, h) in zip(sub.iterrows(),
                                    squarify(sub.Funding_SGDm.values, cx + pad, cy + 2.6,
                                             cw - 2 * pad, ch - 2.6 - pad)):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=CC[c], ec="white", lw=1.5,
                                   alpha=0.45 + 0.5 * r.Faculty / uni.Faculty.max()))
        if w > 6 and h > 3:
            ax.text(x + 0.5, y + 0.8, f"{r.Department}\nS${r.Funding_SGDm:.0f}m",
                    va="top", fontsize=8, color="black")
    ax.add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False, ec="#222222", lw=2.5))
    ax.text(cx + 0.6, cy + 0.4, f"{c}  -  total S${tot:.0f}m", va="top", fontsize=11,
            weight="bold")
ax.set_xlim(0, 100)
ax.set_ylim(60, 0)
ax.axis("off")
ax.set_title("10.2 Treemap - research funding (area) by college > department; "
             "darker = more faculty", weight="bold", loc="left")
fig.tight_layout()
fig.savefig("charts/q2_treemap_funding.png", bbox_inches="tight")
plt.close(fig)

# ---- sunburst ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 9.5))
ax.pie([1], radius=0.25, colors=["#333333"], wedgeprops=dict(width=0.25, edgecolor="white"))
ax.text(0, 0, "University", ha="center", va="center", color="white", fontsize=10, weight="bold")
# equal angle per department -> shows the COUNT of units at each level
lvl_specs = [("College", 0.5), ("School", 0.75), ("Department", 1.0)]
for lvl, r in lvl_specs:
    grp = uni.groupby(["College", "School", "Department"][:["College", "School", "Department"]
                      .index(lvl) + 1], sort=False).size()
    labels = [k if isinstance(k, str) else k[-1] for k in grp.index]
    cols = [CC[k if isinstance(k, str) else k[0]] for k in grp.index]
    alpha_cols = [plt.matplotlib.colors.to_rgba(c, {"College": 1, "School": 0.75,
                                                    "Department": 0.5}[lvl]) for c in cols]
    wedges, _ = ax.pie(grp.values, radius=r, colors=alpha_cols, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.25, edgecolor="white", linewidth=1.2))
    for wdg, lab in zip(wedges, labels):
        ang = np.radians((wdg.theta1 + wdg.theta2) / 2)
        rr = r - 0.125
        rot = np.degrees(ang)
        rot = rot - 180 if 90 < rot % 360 < 270 else rot
        ax.text(rr * np.cos(ang), rr * np.sin(ang), lab, ha="center", va="center",
                rotation=rot if lvl != "College" else 0, fontsize=7 if lvl == "Department" else 8,
                weight="bold" if lvl == "College" else None)
counts = {"Colleges": uni.College.nunique(), "Schools": uni.School.nunique(),
          "Departments": len(uni)}
ax.set_title("10.2 Sunburst - organisational units per hierarchy level\n" +
             ",  ".join(f"{k}: {v}" for k, v in counts.items()), weight="bold")
fig.tight_layout()
fig.savefig("charts/q2_sunburst_levels.png", bbox_inches="tight")
plt.close(fig)

# =============================================================================
# 10.3  DASHBOARD DESIGN PROBLEMS  (bicycle-parts manufacturing dashboard)
# =============================================================================
"""
Evaluated against Few's "Designing dashboards for usability" guidelines:

1. Does not fit on a single screen (Screen fit > overflow).
   The "QA pass rate by model", "Worldwide Warehouse Production by Country"
   and "Choose a date" panels are cut off at the bottom - users must scroll,
   so the whole picture is never visible at once.

2. Visual prominence is not aligned with importance (Arrange data properly /
   highlight important data).
   The most prominent area (top, full width) is the "Choose a part" icon bar -
   a navigation control, not data. Large decorative photo background and heavy
   panel frames draw attention; the KPI that matters (5.25K units produced
   this month) is small and grey.

3. Too many filters / controls taking prime space (Include only what you need).
   Part selector, model checkboxes, on-time slider and date picker occupy the
   whole left column and the top row - about 40% of the screen - while the
   charts are squeezed. It is also unclear which charts each filter affects.

4. Poor colour use (Choose appropriate colours, keep bright colours minimal).
   Six bright, saturated colours for the six models (orange/yellow/blues)
   used on a dark background; similar blues (RDX-EU / RDX-NZ / RDX-USA) are
   hard to tell apart; the map uses yet another, unrelated palette for ~30
   sites (legend truncated with "12 more..."). Colour is not consistent
   across views, and the busy photo background lowers contrast/readability.

5. Ineffective / cluttered chart choices (Support meaningful comparisons).
   * "Production over time" uses horizontal stacked bars per half-month with
     time on the vertical axis - trends over time are hard to read, and only
     the first segment shares a baseline, so models cannot be compared.
   * The map shows sites as tiny dots with truncated labels ("Da Nang ...",
     "Don ...") and a long legend instead of encoding production (size).
   * QA pass rate uses overlapping area shapes on a very narrow y-range
     (0.92-0.93) - exaggerated differences and occlusion.

6. Text is too small / low contrast and truncated (Choose the right text,
   high resolution). Axis labels, legend entries and site names are tiny
   grey-on-dark text, many truncated with "...".

7. Missing context (Supply adequate context).
   No targets, benchmarks, previous-period comparison or units for most
   numbers (e.g. is 5.25K good? what is the QA target?).

8. Weak grouping / layout logic (Colocate items that belong together).
   Filters that act on time (date picker) are far from the time chart; the
   on-time filter is separated from any on-time metric; panel sizes are
   inconsistent, so the reading order is unclear.

Improvements: remove the background image, shrink navigation into a compact
filter bar, put KPI cards (units, on-time %, QA %) with targets at the top,
use a line chart / small multiples for production over time, proportional
symbols on the map, a muted palette with one highlight colour, and fit
everything on one screen.
"""

print("Tutorial 10 charts saved to ./charts/")

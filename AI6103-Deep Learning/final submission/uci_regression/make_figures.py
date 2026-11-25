"""Figures and tables for experiments C, D, E and F (UCI regression part of the report).

Reads the saved runs under results/ (no retraining) and writes
  figures/fig_*.pdf and .png
  tables/tab_*.csv and .tex

  python make_figures.py            # everything
  python make_figures.py c1 d2      # only the named figures / "tables"

Main setting everywhere unless a figure says otherwise: learning rate 1e-3, lambda 0,
no warm-up. Colours: blue = standardised targets (the main HCM setting), orange = raw
targets; in the interval figures blue = HCM, orange = Deep Ensemble, green = MC Dropout.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import rankdata

import analyze_c
import analyze_d
import analyze_e
import analyze_f
import analyze_f_modes

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
SIX = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated: all pairs, light mode
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],   # .ttf only: the PDF backend cannot embed .ttc
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.minor.size": 0, "ytick.minor.size": 0, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 4.5, "pdf.fonttype": 42,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=300)
    plt.close(fig)
    print("wrote figures/%s.pdf/.png" % name)


def save_table(df, name, index=True):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name + ".csv"), index=index)
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write(df.to_latex(index=index, escape=True))
    print("wrote tables/%s.csv/.tex" % name)


def pm(mean, std, fmt="%.3f"):
    return (fmt + " (" + fmt + ")") % (mean, std)


_cache = {}


def frame(which):
    """Per-run data frames of the three analysis scripts (computed once)."""
    if which not in _cache:
        if which == "c":
            df = pd.DataFrame(list(analyze_c.rows()))
        elif which == "d":
            analyze_d.SETS = ["test_s%d" % s for s in range(6)]     # all noise levels
            df = pd.DataFrame(list(analyze_d.rows()))
        else:
            df = pd.DataFrame(list(analyze_e.rows()))
        _cache[which] = df
    return _cache[which]


# ----------------------------------------------------------------------------- C
C_MODES = [("raw", "raw", "o", ORANGE), ("scale:0.1", "y × 0.1", "v", ORANGE),
           ("scale:10", "y × 10", "^", ORANGE), ("center", "centred", "s", BLUE),
           ("standardize", "standardised", "D", BLUE)]
C_NAMES = {"energy2": "energy (2 targets)"}


def fig_c1():
    """Mean ||d_hat|| per dataset and target preprocessing."""
    df = frame("c")
    t = df[df.lr == 1e-3].groupby(["mode", "dataset"])["dnorm"].mean().unstack("dataset")
    order = analyze_c.ORDER
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axvline(1.0, color=MUTED, lw=0.8, zorder=1)
    for mode, label, marker, color in C_MODES:
        ax.plot(t.loc[mode, order].values, range(len(order)), ls="", marker=marker, color=color,
                ms=6, mec="white", mew=0.8, label=label, zorder=3, clip_on=False)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([C_NAMES.get(d, d) for d in order])
    ax.invert_yaxis()
    ax.set_xlim(0.74, 1.06)
    ax.set_xlabel(r"mean $\Vert \hat{d}\Vert _2$ on the clean test set  (1 = on the unit sphere)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.32, 1.0), handletextpad=0.1,
              columnspacing=0.9, title="target preprocessing", title_fontsize=7.5)
    save(fig, "fig_c1_dnorm_by_target")


def violation_by_distance(ds, tag, nb=5):
    """Mean | ||d_hat|| - 1 | per quantile bin of the target's distance from the origin."""
    per_seed = []
    for path in sorted(glob.glob(os.path.join(RES, "stepC", "%s_hcm_%s_lr0.001_lam0_wu0_s*.npz" % (ds, tag)))):
        z = np.load(path)
        t = (z["test_s0_y"] - z["y_shift"]) / z["y_scale"]
        dist = np.sqrt((t ** 2).sum(axis=1))
        dev = np.abs(z["test_s0_dnorm"] - 1.0)
        b = ((rankdata(dist, method="ordinal") - 1) * nb // len(dist)).astype(int)
        per_seed.append([dev[b == i].mean() for i in range(nb)])
    per_seed = np.array(per_seed)
    return per_seed.mean(0), per_seed.std(0)


def fig_c2():
    """Where on the target axis is the unit-norm constraint violated?"""
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.5), sharex=True, sharey=True)
    x = np.arange(1, 6)
    for ax, ds in zip(axes.ravel(), SIX):
        for tag, label, color in (("standardize", "standardised targets", BLUE),
                                  ("raw", "raw targets", ORANGE)):
            m, s = violation_by_distance(ds, tag)
            ax.fill_between(x, np.maximum(m - s, 1e-4), m + s, color=color, alpha=0.15, lw=0)
            ax.plot(x, m, marker="o", color=color, mec="white", mew=0.7, label=label)
        ax.set_yscale("log")
        ax.set_title(ds, loc="left")
        ax.set_xticks(x)
    axes[0, 0].set_ylim(8e-4, 1.0)
    fig.supxlabel("quintile of the target's distance from the origin of the transformed target "
                  "space  (1 = nearest)", fontsize=8, color=INK2, y=0.0)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mean $|\,\Vert \hat{d}\Vert _2 - 1\,|$")
    axes[0, 0].legend(ncol=2, loc="lower left", bbox_to_anchor=(-0.05, 1.18), columnspacing=1.5)
    fig.subplots_adjust(hspace=0.32, wspace=0.12)
    save(fig, "fig_c2_violation_vs_origin")


def fig_c3():
    """Noise detection when only the scale of the target changes."""
    df = frame("c")
    t = df[df.lr == 1e-3].groupby(["mode", "dataset"])["auroc_s5"].mean().unstack("dataset")
    modes, order = ["scale:0.1", "raw", "scale:10"], analyze_c.ORDER
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.text(2.0, 0.515, "chance", ha="right", va="bottom", color=MUTED, fontsize=7.5)
    for ds in order:
        if ds == "kin8nm":
            continue
        ax.plot(range(3), t.loc[modes, ds].values, color=AXIS, lw=1.0, marker="o", ms=3.5, zorder=2)
    y = t.loc[modes, "kin8nm"].values
    ax.plot(range(3), y, color=BLUE, lw=2.0, marker="o", ms=6, mec="white", mew=0.8, zorder=3)
    ax.annotate("kin8nm  %.2f" % y[0], (0, y[0]), xytext=(7, 0), textcoords="offset points",
                va="center", color=INK, fontsize=7.5)
    ax.annotate("%.2f" % y[1], (1, y[1]), xytext=(0, 6), textcoords="offset points",
                ha="center", color=INK, fontsize=7.5)
    yy = t.loc["scale:10", "yacht"]
    ax.annotate("yacht  %.2f" % yy, (2, yy), xytext=(-7, -1), textcoords="offset points",
                ha="right", va="top", color=INK2, fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["y × 0.1", "y (raw)", "y × 10"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(0.25, 1.03)
    ax.set_xlabel("target multiplied by a constant before training")
    ax.set_ylabel("AUROC, clean vs noisy inputs (σ = 5)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, marker="o", mec="white", label="kin8nm"),
                       Line2D([], [], color=AXIS, lw=1, marker="o", ms=3.5, label="the other six datasets")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=1.2)
    save(fig, "fig_c3_target_scale")


# ----------------------------------------------------------------------------- D
D_METHODS = [("hcm_temp", "HCM + temperature (paper)", BLUE),
             ("hcm_cp_const", "HCM + conformal, constant width", BLUE),
             ("hcm_cp_u", "HCM + conformal, width scaled by u", BLUE),
             ("hcm_cp_floor", "HCM + conformal, floored u (ours)", BLUE),
             ("ens_gauss", "Deep Ensemble, Gaussian", ORANGE),
             ("ens_cp_sigma", "Deep Ensemble + conformal", ORANGE),
             ("mcd_gauss", "MC Dropout, Gaussian", AQUA),
             ("mcd_cp_sigma", "MC Dropout + conformal", AQUA)]
P = 0.90


def fig_d1():
    """Clean-test coverage and interval score of every interval method (nominal 90 %)."""
    df = frame("d")
    df = df[np.isclose(df.p, P)]
    panels = [("cov", "test_s0", "coverage, clean test", False),
              ("iscore", "test_s0", "interval score, clean test", True),
              ("iscore", "test_s1", "interval score, noise σ = 1", True)]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True)
    rows = range(len(D_METHODS))
    ticks = {"test_s0": [0.2, 0.5, 1, 2, 5, 10], "test_s1": [5, 10, 20, 50, 100]}
    for ax, (metric, name, title, log) in zip(axes, panels):
        t = df[df.set == name].groupby(["method", "dataset"])[metric].mean().unstack("dataset")[SIX]
        for i, (key, _, color) in enumerate(D_METHODS):
            ax.plot(t.loc[key].values, [i] * 6, ls="", marker="o", ms=3, color=AXIS, zorder=2)
            ax.plot(t.loc[key].mean(), i, ls="", marker="o", ms=6.5, color=color, mec="white",
                    mew=0.8, zorder=3)
        if log:
            ax.set_xscale("log")
            ax.set_xticks(ticks[name])
            ax.set_xticklabels(["%g" % v for v in ticks[name]])
            ax.set_xlabel("in units of std(y), lower is better")
        else:
            ax.axvline(P, color=MUTED, lw=0.8, zorder=1)
            ax.text(P, -0.75, "nominal 0.90", ha="center", va="bottom", color=MUTED, fontsize=7)
            ax.set_xlim(0.45, 1.0)
            ax.set_xlabel("fraction of test targets covered")
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(rows))
    axes[0].set_yticklabels([m[1] for m in D_METHODS])
    axes[0].set_ylim(len(D_METHODS) - 0.5, -0.5)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=6.5, color=INK2, mec="white",
                               label="mean of the six datasets (colour = model)"),
                        Line2D([], [], ls="", marker="o", ms=3, color=AXIS, label="one dataset")],
               loc="lower left", bbox_to_anchor=(0.12, -0.13), ncol=2, columnspacing=1.5)
    fig.subplots_adjust(wspace=0.14)
    save(fig, "fig_d1_intervals_clean")


D_LINES = [("hcm_cp_const", "HCM, constant width", MUTED, "-", "s", 1.2),
           ("hcm_cp_u", "HCM, width scaled by u", BLUE, (0, (4, 2)), "v", 1.2),
           ("hcm_cp_floor", "HCM, floored u (ours)", BLUE, "-", "o", 2.0),
           ("ens_cp_sigma", "Deep Ensemble", ORANGE, "-", "D", 1.5),
           ("mcd_cp_sigma", "MC Dropout", AQUA, "-", "^", 1.5)]


def fig_d2():
    """Conformal intervals under test-time input noise (mean over the six datasets)."""
    df = frame("d")
    df = df[np.isclose(df.p, P)].copy()
    df["sigma"] = df["set"].str[-1].astype(int)
    per_ds = df.groupby(["method", "sigma", "dataset"])[["cov", "iscore"]].mean()
    t = per_ds.groupby(level=[0, 1]).mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for key, label, color, ls, marker, lw in D_LINES:
        for ax, metric in zip(axes, ("cov", "iscore")):
            ax.plot(range(6), t.loc[key][metric].values, color=color, ls=ls, lw=lw, marker=marker,
                    ms=5, mec="white", mew=0.7, label=label)
    axes[0].axhline(P, color=MUTED, lw=0.8, zorder=1)
    axes[0].text(0.35, P + 0.012, "nominal 0.90", ha="left", va="bottom", color=MUTED, fontsize=7.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("coverage")
    axes[1].set_yscale("log")
    axes[1].set_yticks([1, 10, 100])
    axes[1].set_yticklabels(["1", "10", "100"])
    axes[1].set_ylabel("interval score / std(y), lower is better")
    for ax in axes:
        ax.set_xticks(range(6))
        ax.set_xlabel("standard deviation σ of the noise added to test inputs")
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.14, 1.02), columnspacing=1.5,
                   handlelength=2.4, title="split conformal intervals, nominal coverage 0.90",
                   title_fontsize=7.5)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig_d2_intervals_noise")


# ----------------------------------------------------------------------------- E
E_MODES = [("standardize", "standardised targets", BLUE), ("raw", "raw targets", ORANGE)]


def fig_e1():
    """Test error against the learning rate, per dataset."""
    df = frame("e")
    df = df[(df.wu == 0) & (df.lam == 0)]
    g = df.groupby(["mode", "dataset", "lr"])["nrmse"].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.6), sharex=True, sharey=True)
    for ax, ds in zip(axes.ravel(), SIX):
        ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
        for mode, label, color in E_MODES:
            s = g.loc[mode, ds]
            lr, mu, sd = s.index.values, s["mean"].values, s["std"].values
            ax.fill_between(lr, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
            ax.plot(lr, mu, marker="o", color=color, mec="white", mew=0.7, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ds, loc="left")
    axes[0, 0].text(1e-2, 1.06, "predicting the training mean", ha="right", va="bottom",
                    color=MUTED, fontsize=7)
    axes[0, 0].set_ylim(0.04, 2.6)
    axes[0, 0].set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    axes[0, 0].set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    for ax in axes[1]:
        ax.set_xticks([1e-4, 1e-3, 1e-2])
        ax.set_xticklabels(["1e-4\n(paper)", "1e-3\n(ours)", "1e-2"])
        ax.set_xlabel("learning rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("test RMSE / std(y)")
    axes[0, 0].legend(ncol=2, loc="lower left", bbox_to_anchor=(-0.05, 1.18), columnspacing=1.5)
    fig.subplots_adjust(hspace=0.32, wspace=0.12)
    save(fig, "fig_e1_learning_rate")


def fig_e2():
    """Effect of the norm penalty lambda at learning rate 1e-3."""
    df = frame("e")
    df = df[(df.wu == 0) & (df.lr == 1e-3)]
    lams = sorted(df.lam.unique())
    panels = [("dnorm", r"mean $\Vert \hat{d}\Vert _2$", None),
              ("nrmse", "test RMSE relative to λ = 0", "ratio"),
              ("auroc_s1", "noise-detection AUROC (σ = 1)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    for ax, (metric, title, kind) in zip(axes, panels):
        for mode, label, color in E_MODES:
            t = df[df["mode"] == mode].groupby(["dataset", "lam"])[metric].mean().unstack("lam")[lams]
            if kind == "ratio":
                t = t.div(t[0.0], axis=0)
            for ds in SIX:
                ax.plot(range(len(lams)), t.loc[ds].values, color=color, lw=0.7, alpha=0.3, zorder=2)
            ax.plot(range(len(lams)), t.loc[SIX].mean().values, color=color, lw=2.0, marker="o",
                    mec="white", mew=0.7, label=label, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(lams)))
        ax.set_xticklabels(["%g" % v for v in lams])
        ax.set_xlabel("norm-penalty weight λ")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK2, lw=0.7, alpha=0.5))
    labels.append("thin lines: single datasets")
    axes[0].legend(handles, labels, ncol=3, loc="lower left", bbox_to_anchor=(-0.12, 1.14),
                   columnspacing=1.5)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig_e2_lambda")


# ----------------------------------------------------------------------------- tables
PAPER_T11 = {"wine": (.554, .550, .552, .554), "concrete": (4.31, 4.30, 4.53, 4.64),
             "energy": (1.82, 2.71, 2.38, 2.06), "kin8nm": (.066, .064, .071, 1.377),
             "power": (3.47, 4.44, 5.14, 4.58), "yacht": (.053, .054, .048, .060)}


def tables():
    # C: target preprocessing, lr 1e-3, mean (std) over 5 seeds
    c = frame("c")
    c = c[c.lr == 1e-3]
    blocks = []
    for metric, name, fmt in (("rmse", "test RMSE", "%.3g"), ("dnorm", "mean ||d_hat||", "%.3f"),
                              ("auroc_s5", "AUROC sigma=5", "%.3f"),
                              ("sp_clean", "Spearman(u, error), clean", "%.2f")):
        g = c.groupby(["mode", "dataset"])[metric].agg(["mean", "std"])
        cell = g.apply(lambda r: pm(r["mean"], r["std"], fmt), axis=1).unstack("dataset")
        cell = cell.loc[analyze_c.MODES, analyze_c.ORDER]
        cell.index = pd.MultiIndex.from_product([[name], cell.index], names=["metric", "target"])
        blocks.append(cell)
    save_table(pd.concat(blocks), "tab_c_target_preprocessing")

    # D: interval methods, nominal 0.90, mean over the six datasets (20 seeds each)
    d = frame("d")
    d = d[np.isclose(d.p, P)]
    per_ds = d.groupby(["method", "set", "dataset"])[["cov", "width", "iscore"]].mean()
    m = per_ds.groupby(level=[0, 1]).mean()
    out = pd.DataFrame({
        "coverage clean": m.xs("test_s0", level="set")["cov"],
        "width clean": m.xs("test_s0", level="set")["width"],
        "int. score clean": m.xs("test_s0", level="set")["iscore"],
        "coverage s=1": m.xs("test_s1", level="set")["cov"],
        "int. score s=1": m.xs("test_s1", level="set")["iscore"],
        "coverage s=5": m.xs("test_s5", level="set")["cov"],
        "int. score s=5": m.xs("test_s5", level="set")["iscore"]})
    out = out.loc[[k for k in out.index if not k.startswith("hcmraw")]]
    for col in out:
        out[col] = out[col].map(lambda v: ("%.3f" if "coverage" in col else "%.2f") % v)
    save_table(out, "tab_d_intervals")

    g = d[d.set == "test_s0"].groupby(["method", "dataset"])["cov"].agg(["mean", "std"])
    cov = g.apply(lambda r: pm(r["mean"], r["std"]), axis=1).unstack("dataset")[SIX]
    save_table(cov.loc[[k for k in cov.index if not k.startswith("hcmraw")]], "tab_d_coverage_by_dataset")

    rmse = []
    for path in glob.glob(os.path.join(RES, "stepD", "*.json")):
        j = json.load(open(path))
        rmse.append(dict(model="%s (%s targets)" % (j["method"], j["target_mode"]),
                         dataset=j["dataset"], rmse=j["rmse_test_s0"]))
    g = pd.DataFrame(rmse).groupby(["model", "dataset"])["rmse"].agg(["mean", "std"])
    save_table(g.apply(lambda r: pm(r["mean"], r["std"], "%.3g"), axis=1).unstack("dataset")[SIX],
               "tab_d_test_rmse")

    # E: sensitivity
    e = frame("e")
    base = e[(e.wu == 0) & (e.lam == 0)]
    g = base.groupby(["mode", "lr", "dataset"])["nrmse"].agg(["mean", "std"])
    cell = g.apply(lambda r: pm(r["mean"], r["std"]), axis=1).unstack("dataset")[SIX]
    cell["mean of six"] = g["mean"].unstack("dataset")[SIX].mean(axis=1).map(lambda v: "%.3f" % v)
    save_table(cell, "tab_e_learning_rate")

    lam = e[(e.wu == 0) & (e.lr == 1e-3)]
    cols = ["dnorm", "nrmse", "auroc_s1", "auroc_s5", "sp_clean", "sp_pooled"]
    t = lam.groupby(["mode", "lam", "dataset"])[cols].mean().groupby(level=[0, 1]).mean()
    save_table(t.applymap(lambda v: "%.3f" % v), "tab_e_lambda")

    wu = e[e.lam == 0]
    t = wu.groupby(["mode", "lr", "wu", "dataset"])[["nrmse", "sp_pooled", "auroc_s1"]].mean() \
          .groupby(level=[0, 1, 2]).mean().unstack("wu")
    save_table(t.applymap(lambda v: "%.3f" % v), "tab_e_warmup")

    paper = e[(e["mode"] == "raw") & (e.lr == 1e-4) & (e.wu == 0)]
    ours = paper.groupby(["dataset", "lam"])["mae_val"].mean().unstack("lam")
    small = {}
    for b in (8, 1):
        for path in glob.glob(os.path.join(RES, "stepE_bs%d" % b, "*.npz")):
            z = np.load(path)
            ds = os.path.basename(path).split("_")[0]
            small.setdefault((ds, b), []).append(float(np.abs(z["val_y"] - z["val_yhat"]).mean()))
    rows = []
    for ds in SIX:
        row = {"dataset": ds}
        for i, l in enumerate((0.0, 1.0, 3.0, 5.0)):
            row["paper lam=%g" % l] = "%.3f" % PAPER_T11[ds][i]
        for l in (0.0, 1.0, 3.0, 5.0):
            row["ours lam=%g" % l] = "%.3f" % ours.loc[ds, l]
        for b in (8, 1):
            v = small.get((ds, b))
            row["ours lam=0, batch %d" % b] = "%.3f" % np.mean(v) if v else "-"
        rows.append(row)
    save_table(pd.DataFrame(rows), "tab_e_paper_table11", index=False)


# ----------------------------------------------------------------------------- F
F_LINES = [("hcm", "HCM, floored u", BLUE, (0, (1.2, 1.6)), "o", 1.3),
           ("rr", "residual-origin score only", BLUE, (0, (4, 2)), "v", 1.3),
           ("max", "both scores, max rule (ours)", BLUE, "-", "s", 2.0),
           ("ens", "Deep Ensemble", ORANGE, "-", "D", 1.5),
           ("mcd", "MC Dropout", AQUA, "-", "^", 1.5)]
F_BRANCH = "detach"       # the single-trunk variant is the one shown in the figure


def _f_frames():
    if "f" not in _cache:
        df = analyze_f.frame()
        comb = pd.DataFrame(list(analyze_f.combined_rows()))
        _cache["f"] = (df, comb)
    return _cache["f"]


def _f_curves():
    """Interval score and coverage for sigma = 0..5, mean of six datasets, per line of F_LINES."""
    df, comb = _f_frames()

    def mean6(d, col):
        return d.groupby("dataset")[col].mean().reindex(SIX).mean()
    c = comb[comb.branch == F_BRANCH]
    src = {"hcm": (df[df.name == "hcm [u]"], "floor"), "ens": (df[df.name == "ens [sigma]"], "prop"),
           "mcd": (df[df.name == "mcd [sigma]"], "prop"),
           "rr": (c[c.rule == "R_r"], None), "max": (c[c.rule == "max"], None)}
    out = {}
    for key, (d, rule) in src.items():
        tag = (lambda m, k: "%s_%s_s%d" % (m, rule, k)) if rule else (lambda m, k: "%s_s%d" % (m, k))
        out[key] = {m: [mean6(d, tag(m, k)) for k in range(6)] for m in ("is", "cov")}
    return out


def fig_f1():
    """The two changes of experiment F: accuracy, clean intervals, intervals under noise."""
    df, comb = _f_frames()
    cur = _f_curves()
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.55), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    # (a) accuracy relative to the plain MLP
    one = df.drop_duplicates(["model", "dataset", "seed"])
    acc = [("hcm", "HCM"), ("hcmp-shared", "+ prediction head,\nshared trunk"),
           ("hcmp-detach", "+ prediction head,\ndetached (ours)")]
    ax = axes[0]
    for i, (model, _) in enumerate(acc):
        t = one[one.model == model].groupby("dataset")["rmse_ratio"].mean().reindex(SIX)
        ax.plot(t.values, [i] * 6, ls="", marker="o", ms=3, color=AXIS, zorder=2)
        ax.plot(t.mean(), i, ls="", marker="o", ms=6.5, color=BLUE, mec="white", mew=0.8, zorder=3)
        ax.text(t.mean(), i - 0.24, "%.2f" % t.mean(), ha="center", va="bottom", fontsize=7.5)
    ax.axvline(1.0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(range(len(acc)))
    ax.set_yticklabels([a[1] for a in acc])
    ax.set_ylim(len(acc) - 0.5, -0.6)
    ax.set_xlim(0.9, 1.75)
    ax.set_xticks([1.0, 1.25, 1.5, 1.75])
    ax.set_xlabel("test RMSE / RMSE of plain MLP")
    ax.set_title("accuracy", loc="left", pad=8)
    ax.grid(axis="y", visible=False)

    # (b) interval score on clean data
    ax = axes[1]
    for i, (key, label, color, _, marker, _) in enumerate(F_LINES):
        v = cur[key]["is"][0]
        ax.plot(v, i, ls="", marker=marker, ms=6.5, color=color, mec="white", mew=0.8, zorder=3)
        ax.text(v, i - 0.24, "%.2f" % v, ha="center", va="bottom", fontsize=7.5)
    ax.set_yticks(range(len(F_LINES)))
    ax.set_yticklabels(["HCM, floored u", "residual-origin\nscore only", "both scores,\nmax rule (ours)",
                        "Deep Ensemble", "MC Dropout"])
    ax.set_ylim(len(F_LINES) - 0.5, -0.6)
    ax.set_xlim(1.1, 1.56)
    ax.set_xlabel("interval score / std(y)")
    ax.set_title("clean test, nominal 0.90", loc="left", pad=8)
    ax.grid(axis="y", visible=False)

    # (c) interval score under input noise
    ax = axes[2]
    for key, label, color, ls, marker, lw in F_LINES:
        ax.plot(range(6), cur[key]["is"], color=color, ls=ls, lw=lw, marker=marker, ms=4.5,
                mec="white", mew=0.7, label=label)
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_xticks(range(6))
    ax.set_xlabel("std σ of the noise added to test inputs")
    ax.set_title("interval score under input noise", loc="left", pad=8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.27), ncol=3, handlelength=2.6,
              columnspacing=1.4, labelspacing=0.35, borderaxespad=0)
    axes[0].legend(handles=[Line2D([], [], ls="", marker="o", ms=6.5, color=BLUE, mec="white",
                                   label="mean of six datasets"),
                            Line2D([], [], ls="", marker="o", ms=3, color=AXIS, label="one dataset")],
                   loc="upper left", bbox_to_anchor=(-0.9, -0.27), ncol=1, labelspacing=0.35,
                   borderaxespad=0)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "fig_f1_changes")


def tables_f():
    df, comb = _f_frames()

    def mean6(d, col):
        return d.groupby("dataset")[col].mean().reindex(SIX).mean()
    one = df.drop_duplicates(["model", "dataset", "seed"])

    def ratio(model):
        return mean6(one[one.model == model], "rmse_ratio")
    rows = []

    def add(label, model, d, pre, score_d=None):
        sd = d if score_d is None else score_d
        rows.append({"method": label, "RMSE / MLP": "%.3f" % ratio(model),
                     "Spearman(score, error)": "%.3f" % mean6(sd, "spearman"),
                     "AUROC s=5": "%.3f" % mean6(sd, "auroc_s5"),
                     "coverage clean": "%.3f" % mean6(d, "cov_%ss0" % pre),
                     "IS clean": "%.2f" % mean6(d, "is_%ss0" % pre),
                     "IS s=1": "%.1f" % mean6(d, "is_%ss1" % pre),
                     "IS s=5": "%.1f" % mean6(d, "is_%ss5" % pre)})
    add("HCM, floored u_t (Sec. 2.4)", "hcm", df[df.name == "hcm [u]"], "floor_")
    for branch in ("detach", "own"):
        c = comb[comb.branch == branch]
        u_t = df[df.name == "hcmp-%s [u]" % branch]
        add("%s: floored u_t" % branch, "hcmp-" + branch, c[c.rule == "floor(u_t)"], "", score_d=u_t)
        add("%s: residual-origin R_r" % branch, "rhcm-" + branch, c[c.rule == "R_r"], "")
        add("%s: both, product rule" % branch, "rhcm-" + branch, c[c.rule == "product"], "")
        add("%s: both, max rule" % branch, "rhcm-" + branch, c[c.rule == "max"], "")
    add("Deep Ensemble, conformal", "ens", df[df.name == "ens [sigma]"], "prop_")
    add("MC Dropout, conformal", "mcd", df[df.name == "mcd [sigma]"], "prop_")
    save_table(pd.DataFrame(rows), "tab_f_methods", index=False)

    acc = one.groupby(["model", "dataset"], sort=False)["rmse_ratio"].mean().unstack("dataset")[SIX]
    acc["mean of six"] = acc.mean(axis=1)
    save_table(acc.applymap(lambda v: "%.3f" % v), "tab_f_rmse_ratio")

    modes = analyze_f_modes.frame()
    cell = modes.groupby(["model", "mode", "dataset"], sort=False).mean(numeric_only=True).drop(columns="seed")
    t = cell.groupby(["model", "mode"], sort=False).mean()
    save_table(t.applymap(lambda v: "%.3f" % v), "tab_f_target_modes")


FIGS = {"c1": fig_c1, "c2": fig_c2, "c3": fig_c3, "d1": fig_d1, "d2": fig_d2,
        "e1": fig_e1, "e2": fig_e2, "f1": fig_f1, "tables": tables, "tables_f": tables_f}

if __name__ == "__main__":
    for key in (sys.argv[1:] or list(FIGS)):
        FIGS[key]()

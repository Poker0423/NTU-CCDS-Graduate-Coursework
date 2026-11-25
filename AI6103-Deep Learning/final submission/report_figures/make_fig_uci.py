"""Figures 3 and 4 of the report, redrawn at column width in the paper's font.
Reads the saved runs through hcm_uci/make_figures.py; no numbers are changed."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "uci_regression"))
import make_figures as mf
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
    "ytick.color": "black", "axes.edgecolor": "0.4", "axes.linewidth": 0.5,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "savefig.bbox": None, "savefig.pad_inches": 0,
})
W = 3.3


def fig3():
    fig, axes = plt.subplots(2, 3, figsize=(W, 1.47), sharex=True, sharey=True)
    x = np.arange(1, 6)
    for ax, ds in zip(axes.ravel(), mf.SIX):
        for tag, label, color in (("standardize", "Standardized targets", mf.BLUE),
                                  ("raw", "Raw targets", mf.ORANGE)):
            m, s = mf.violation_by_distance(ds, tag)
            ax.fill_between(x, np.maximum(m - s, 1e-4), m + s, color=color, alpha=0.15, lw=0)
            ax.plot(x, m, marker="o", color=color, mec="white", mew=0.5, label=label)
        ax.set_yscale("log")
        ax.text(0.5, 0.96, ds, transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_xticks(x)
        ax.tick_params(pad=1.5)
    axes[0, 0].set_ylim(8e-4, 6.0)
    axes[0, 0].set_yticks([1e-3, 1e-2, 1e-1, 1])
    axes[0, 0].set_yticklabels([r"$10^{-3}$", "", r"$10^{-1}$", ""])
    fig.supxlabel("Quintile of the target's distance from the origin (1 = nearest)",
                  fontsize=8, y=0.01)
    fig.supylabel(r"Mean $|\,\Vert\hat{d}\Vert_2 - 1\,|$", fontsize=8, x=0.005)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.56, 1.01),
               columnspacing=1.5, handlelength=1.8, borderaxespad=0)
    fig.subplots_adjust(left=0.165, right=0.995, top=0.875, bottom=0.235, hspace=0.12, wspace=0.08)
    fig.savefig(os.path.join(HERE, "fig_origin.pdf"))
    fig.savefig(os.path.join(HERE, "fig_origin_preview.png"), dpi=400)


def fig4():
    df, comb = mf._f_frames()
    cur = mf._f_curves()
    fig = plt.figure(figsize=(W, 1.68))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 5], width_ratios=[1, 1.15],
                          left=0.25, right=0.99, top=0.915, bottom=0.175, hspace=1.05, wspace=0.28)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    one = df.drop_duplicates(["model", "dataset", "seed"])
    acc = [("hcm", "HCM"), ("hcmp-shared", "+ head, shared"), ("hcmp-detach", "+ head, detached")]
    for i, (model, _) in enumerate(acc):
        t = one[one.model == model].groupby("dataset")["rmse_ratio"].mean().reindex(mf.SIX)
        a.plot(t.values, [i] * 6, ls="", marker="o", ms=2.2, color="0.7", zorder=2)
        a.plot(t.mean(), i, ls="", marker="o", ms=5, color=mf.BLUE, mec="white", mew=0.6, zorder=3)
        a.text(2.06, i, "%.2f" % t.mean(), ha="right", va="center", fontsize=7.5)
    a.axvline(1.0, color="0.5", lw=0.6, zorder=1)
    a.set_yticks(range(3)); a.set_yticklabels([x[1] for x in acc])
    a.set_ylim(2.5, -0.5); a.set_xlim(0.93, 2.08); a.set_xticks([1.0, 1.25, 1.5, 1.75])
    a.set_xlabel("RMSE / RMSE of MLP", labelpad=1)
    a.grid(axis="y", visible=False)

    names = ["HCM, floored $u$", "Residual-origin", "Both, max (ours)", "Deep Ensemble", "MC Dropout"]
    for i, (key, _, color, _, marker, _) in enumerate(mf.F_LINES):
        v = cur[key]["is"][0]
        b.plot(v, i, ls="", marker=marker, ms=5, color=color, mec="white", mew=0.6, zorder=3)
        b.text(1.735, i, "%.2f" % v, ha="right", va="center", fontsize=7.5)
    b.set_yticks(range(5)); b.set_yticklabels(names)
    b.set_ylim(4.5, -0.5); b.set_xlim(1.1, 1.75); b.set_xticks([1.2, 1.4])
    b.set_xlabel("Clean interval score", labelpad=1)
    b.grid(axis="y", visible=False)

    for key, _, color, ls, marker, lw in mf.F_LINES:
        c.plot(range(6), cur[key]["is"], color=color, ls=ls, lw=lw * 0.8, marker=marker,
               ms=3.5, mec="white", mew=0.5)
    c.set_yscale("log"); c.set_yticks([1, 10, 100]); c.set_yticklabels(["1", "10", "100"])
    c.set_xticks(range(6))
    c.set_xlabel(r"Input noise std $\sigma$", labelpad=1)
    c.set_title("Interval score under noise", pad=3)
    c.grid(axis="x", visible=False)
    for ax in (a, b, c):
        ax.tick_params(pad=1.5)
    fig.savefig(os.path.join(HERE, "fig_changes.pdf"))
    fig.savefig(os.path.join(HERE, "fig_changes_preview.png"), dpi=400)


if __name__ == "__main__":
    fig3(); fig4()

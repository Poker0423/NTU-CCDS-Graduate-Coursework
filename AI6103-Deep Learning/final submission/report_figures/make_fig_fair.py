"""Figure 2: OOD detection under matched training (CIFAR-10).
Means and AUROC SDs are the values reported in the text. Three FPR@95 SDs
(MSP in both settings, MDS without MixUp) and the one-decimal FPR means of
MSP/MDS were read from Role 2's original plot, since the raw numbers are
not in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 0, "pdf.fonttype": 42,
})
methods = ["HCM", "MSP", "MDS", "KNN", "Ensemble"]
colors = ["#6f4e9e", "#579dcb", "#9fcae1", "#dc6f5e", "#f4ad8a"]
# (mean, sd); sd None = single ensemble result
data = {
    ("AUROC", "No MixUp"): [(83.04, 0.29), (88.93, 0.31), (72.90, 2.38), (92.01, 0.36), (90.90, None)],
    ("AUROC", "Pairwise MixUp"): [(84.90, 1.27), (84.82, 1.29), (90.87, 0.23), (92.75, 0.85), (90.33, None)],
    ("FPR@95", "No MixUp"): [(59.94, 1.76), (64.6, 1.3), (70.6, 1.8), (42.36, 2.62), (55.40, None)],
    ("FPR@95", "Pairwise MixUp"): [(49.88, 2.17), (54.3, 2.6), (40.39, 1.01), (39.34, 6.82), (45.62, None)],
}
fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.5), sharey=True)
for i, metric in enumerate(["AUROC", "FPR@95"]):
    for j, setting in enumerate(["No MixUp", "Pairwise MixUp"]):
        ax = axes[i, j]
        vals = data[(metric, setting)]
        y = range(len(methods))
        m = [v[0] for v in vals]
        s = [v[1] or 0 for v in vals]
        ax.barh(y, m, height=0.68, color=colors, zorder=2)
        k = [n for n, v in enumerate(vals) if v[1]]
        ax.errorbar([m[n] for n in k], k, xerr=[s[n] for n in k], fmt="none",
                    ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
        for yy, mm, ss in zip(y, m, s):
            ax.text(mm + ss + 1.5, yy, f"{mm:.1f}", va="center", ha="left", fontsize=7.5)
        ax.set_yticks(list(y)); ax.set_yticklabels(methods)
        ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color="0.9", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        arrow = r"$\uparrow$" if metric == "AUROC" else r"$\downarrow$"
        ax.set_xlabel(f"{metric} (%) {arrow}", labelpad=1)
        if i == 0:
            ax.set_title(setting, pad=3)
        ax.tick_params(axis="x", pad=1.5)
axes[0, 0].invert_yaxis()
fig.subplots_adjust(left=0.165, right=0.995, top=0.94, bottom=0.115, wspace=0.06, hspace=0.42)
fig.savefig("fig_fair_comparison.pdf")
fig.savefig("fig_fair_comparison_preview.png", dpi=400)

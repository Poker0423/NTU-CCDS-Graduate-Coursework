"""Test-accuracy curves (mean over seeds, band = ±1 std) for every config group.

    python plot_curves.py --runs_dir runs --out results/test_acc_curves.png

Colour = training loss (CE blue, HCM orange); line style = MixUp (solid none, dashed k=2
soft, dash-dot k=2 decomp, dotted k>2). A second panel shows the mean ||d|| of HCM runs on the test set.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR = {"ce": "#2a78d6", "hcm": "#eb6834"}


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    def col(k):
        return np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
    return col("epoch"), col("test_acc"), col("mean_dnorm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out", default="results/test_acc_curves.png")
    a = p.parse_args()
    groups = defaultdict(list)
    for cfgp in glob.glob(os.path.join(a.runs_dir, "*", "config.json")):
        with open(cfgp) as f:
            cfg = json.load(f)
        logp = os.path.join(os.path.dirname(cfgp), "log.csv")
        if os.path.exists(logp):
            groups[cfg["group"]].append((cfg, read_log(logp)))
    if not groups:
        print("no runs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g in sorted(groups):
        cfg = groups[g][0][0]
        n = min(len(e) for _, (e, _, _) in groups[g])
        if n == 0:
            continue
        ep = groups[g][0][1][0][:n]
        acc = np.stack([lg[1][:n] for _, lg in groups[g]])
        ls = "-" if cfg["mix_k"] < 2 else ("--" if cfg["mix_k"] == 2 else ":")
        if cfg.get("mix_target") == "decomp" and cfg["mix_k"] >= 2:
            ls = "-."
        c = COLOR[cfg["method"]]
        m, s = acc.mean(0), acc.std(0)
        axes[0].plot(ep, m, ls, color=c, lw=2, label=f"{g} (n={len(acc)})")
        axes[0].fill_between(ep, m - s, m + s, color=c, alpha=0.15, lw=0)
        if cfg["method"] == "hcm":
            dn = np.stack([lg[2][:n] for _, lg in groups[g]])
            axes[1].plot(ep, dn.mean(0), ls, color=c, lw=2, label=g)
    axes[0].set(xlabel="epoch", ylabel="CIFAR-10 test accuracy (%)", title="Test accuracy")
    axes[1].set(xlabel="epoch", ylabel="mean ||d|| on test set", title="HCM direction norm")
    axes[1].axhline(1.0, color="#888888", lw=1)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.lines:
            ax.legend(fontsize=8, frameon=False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()

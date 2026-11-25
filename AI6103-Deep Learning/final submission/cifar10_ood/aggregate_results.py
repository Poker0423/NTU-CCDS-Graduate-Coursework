"""Collect metrics.json (+ ood_hcm.json if present) from all runs into summary tables.

    python aggregate_results.py --runs_dir runs --out_dir results

Writes
    results/all_runs.csv          one row per run
    results/accuracy_summary.csv  one row per config group: mean / std / n over seeds
    results/accuracy_summary.md   the same as a Markdown table (paste into the report)
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

ORDER = ["ce", "hcm", "ce_mix-k2-a1", "hcm_mix-k2-a1-decomp", "hcm_mix-k2-a1", "hcm_mix-k20-a0.5", "ce_mix-k20-a0.5"]


def load_runs(runs_dir):
    rows = []
    for m in sorted(glob.glob(os.path.join(runs_dir, "*", "metrics.json"))):
        with open(m) as f:
            r = json.load(f)
        ood = os.path.join(os.path.dirname(m), "ood_hcm.json")
        if os.path.exists(ood):
            with open(ood) as f:
                o = json.load(f).get("paper_eq2", {})
            for grp in ("_near", "_far", "_avg"):
                if grp in o:
                    r[f"auroc{grp}"] = 100 * o[grp]["auroc"]
                    r[f"fpr95{grp}"] = 100 * o[grp]["fpr95"]
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out_dir", default="results")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = load_runs(a.runs_dir)
    if not rows:
        print("no finished runs found")
        return

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out_dir, "all_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metrics = ["test_acc", "test_acc_dir", "test_nll", "mean_dnorm", "mean_R",
               "auroc_avg", "fpr95_avg", "auroc_near", "auroc_far", "train_time_min"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    order = sorted(groups, key=lambda g: (ORDER.index(g) if g in ORDER else len(ORDER), g))

    summary = []
    for g in order:
        rs = groups[g]
        s = {"group": g, "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs),
             "n_diverged": sum(bool(r.get("diverged")) for r in rs)}
        for m in metrics:
            v = [r[m] for r in rs if m in r and r[m] is not None and np.isfinite(r[m])]
            if v:
                s[f"{m}_mean"], s[f"{m}_std"] = float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        summary.append(s)

    skeys = ["group", "n_seeds", "seeds", "n_diverged"] + [f"{m}_{x}" for m in metrics for x in ("mean", "std")]
    with open(os.path.join(a.out_dir, "accuracy_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=skeys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    def cell(s, m, nd=2):
        return f"{s[m+'_mean']:.{nd}f} ± {s[m+'_std']:.{nd}f}" if f"{m}_mean" in s else "–"

    lines = ["| Config | Seeds | Test acc (%) | HCM acc argmax d (%) | HCM OOD AUROC avg (%) | HCM FPR@95 avg (%) |",
             "|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['group']} | {s['n_seeds']} | {cell(s,'test_acc')} | {cell(s,'test_acc_dir')} | "
                     f"{cell(s,'auroc_avg')} | {cell(s,'fpr95_avg')} |")
    md = "\n".join(lines) + "\n\nmean ± sample std over seeds; last-epoch model; CIFAR-10 test set (10k).\n"
    with open(os.path.join(a.out_dir, "accuracy_summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()

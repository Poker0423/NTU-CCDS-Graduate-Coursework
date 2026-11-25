"""Experiment C: how target preprocessing changes the HCM uncertainty score.

Reads results/stepC/*.json and *.npz (no retraining) and reports, per learning rate,
target mode and dataset, the mean and std over seeds of:

  rmse            test RMSE on clean inputs, original target units
  sp_clean        Spearman(u, per-sample error) on the clean test set
  sp_pooled       the same, pooling the clean and all noisy test sets
  auroc_s1/_s5    can u separate clean test inputs from inputs with N(0, sigma^2) noise
  dnorm           mean ||d_hat|| on the clean test set (1 = on the unit sphere)
  sp_dev_origin   Spearman(| ||d_hat|| - 1 |, -||t||): is the norm violation larger for
                  targets close to the origin of the TRANSFORMED target space?
  sp_u_origin     Spearman(u, -||t||): the same for the full score u = |R_hat| * violation

  python analyze_c.py            # prints tables, writes results/stepC/summary.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "results", "stepC")
ORDER = ["wine", "concrete", "energy", "kin8nm", "power", "yacht", "energy2"]
MODES = ["raw", "scale:0.1", "scale:10", "center", "standardize"]
METRICS = ["rmse", "sp_clean", "sp_pooled", "auroc_s1", "auroc_s5", "dnorm",
           "sp_dev_origin", "sp_u_origin"]


def rows():
    for jpath in sorted(glob.glob(os.path.join(EXP, "*.json"))):
        m = json.load(open(jpath))
        z = np.load(jpath[:-5] + ".npz")
        y, R, dn = z["test_s0_y"], z["test_s0_R"], z["test_s0_dnorm"]
        t = (y - z["y_shift"]) / z["y_scale"]            # transformed target
        dist = np.sqrt((t ** 2).sum(axis=1))             # distance from the origin
        dev = np.abs(dn - 1.0)
        u = np.abs(R) * dev
        ok = np.isfinite(u).all() and np.isfinite(m["rmse_test_s0"])
        yield dict(dataset=m["dataset"], mode=m["target_mode"], lr=m["lr"], seed=m["seed"],
                   finite=ok,
                   rmse=m["rmse_test_s0"], sp_clean=m["spearman_clean"],
                   sp_pooled=m["spearman_pooled"], auroc_s1=m["auroc_s1"],
                   auroc_s5=m["auroc_s5"], dnorm=float(dn.mean()),
                   sp_dev_origin=float(spearmanr(dev, -dist).correlation) if ok else np.nan,
                   sp_u_origin=float(spearmanr(u, -dist).correlation) if ok else np.nan)


if __name__ == "__main__":
    df = pd.DataFrame(list(rows()))
    print("runs: %d, non-finite: %d, seeds per cell: %s" % (
        len(df), int((~df["finite"]).sum()), sorted(df.groupby(["dataset", "mode", "lr"]).size().unique())))
    g = df.groupby(["lr", "mode", "dataset"])[METRICS].agg(["mean", "std"])
    g.to_csv(os.path.join(EXP, "summary.csv"))
    pd.set_option("display.width", 250)
    for lr in sorted(df["lr"].unique(), reverse=True):
        for metric in METRICS:
            sub = g.loc[lr][metric]
            cell = sub.apply(lambda r: "%.3g (%.2g)" % (r["mean"], r["std"]), axis=1).unstack("dataset")
            print("\n== lr %g | %s   mean (std) over seeds" % (lr, metric))
            print(cell.loc[MODES, ORDER].to_string())

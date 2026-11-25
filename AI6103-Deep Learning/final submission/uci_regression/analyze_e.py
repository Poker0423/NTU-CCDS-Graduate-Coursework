"""Experiment E: sensitivity of HCM to the learning rate, lambda_norm and warm-up.

Reads results/stepE/*.json and *.npz (no retraining). For every run it records both the
prediction error and the quality of the uncertainty score (the paper's lambda ablation,
appendix A.8, reports validation MAE only):

  nrmse       clean-test RMSE / std(y_train)   (comparable across datasets)
  mae_val     validation MAE, original units   (the paper's metric)
  sp_clean    Spearman(u, error), clean test
  sp_pooled   Spearman(u, error), clean + noisy test sets pooled
  auroc_s1/5  clean vs noisy inputs (sigma = 1, 5) separated by u
  dnorm       mean ||d_hat|| on the clean test set

  python analyze_e.py          # prints tables, writes results/stepE/summary.csv
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "results", "stepE")
ORDER = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]
METRICS = ["nrmse", "mae_val", "sp_clean", "sp_pooled", "auroc_s1", "auroc_s5", "dnorm"]


def rows():
    for jpath in sorted(glob.glob(os.path.join(EXP, "*.json"))):
        m = json.load(open(jpath))
        z = np.load(jpath[:-5] + ".npz")
        div = bool(m.get("diverged", False))
        yield dict(dataset=m["dataset"], mode=m["target_mode"], lr=m["lr"], lam=m["lam"],
                   wu=m["warmup_epochs"], seed=m["seed"], diverged=div,
                   nrmse=m["rmse_test_s0"] / float(z["train_y"].std()),
                   mae_val=float(np.abs(z["val_y"] - z["val_yhat"]).mean()),
                   sp_clean=m.get("spearman_clean", np.nan),
                   sp_pooled=m.get("spearman_pooled", np.nan),
                   auroc_s1=m.get("auroc_s1", np.nan), auroc_s5=m.get("auroc_s5", np.nan),
                   dnorm=m.get("dnorm_mean_s0", np.nan))


def over_datasets(df, by, metric):
    """Mean over seeds per dataset first, then mean over the six datasets."""
    per_ds = df.groupby(by + ["dataset"])[metric].mean()
    return per_ds.groupby(level=list(range(len(by)))).mean()


if __name__ == "__main__":
    df = pd.DataFrame(list(rows()))
    df.groupby(["mode", "lr", "lam", "wu", "dataset"])[METRICS].agg(["mean", "std"]) \
      .to_csv(os.path.join(EXP, "summary.csv"))
    pd.set_option("display.width", 250)
    print("runs: %d, diverged: %d, seeds per cell: %s" % (
        len(df), int(df["diverged"].sum()),
        sorted(df.groupby(["mode", "lr", "lam", "wu", "dataset"]).size().unique())))

    for mode in ["standardize", "raw"]:
        d = df[df["mode"] == mode]
        grid = d[d.wu == 0]
        print("\n" + "=" * 30, "target mode:", mode, "=" * 30)
        for metric in ["nrmse", "sp_clean", "sp_pooled", "auroc_s1", "auroc_s5", "dnorm"]:
            t = over_datasets(grid, ["lr", "lam"], metric).unstack("lam")
            print("\n-- %s : rows lr, columns lambda (warm-up 0), mean over 6 datasets" % metric)
            print(t.round(3).to_string())
        print("\n-- warm-up epochs (lambda 0): rows lr, columns warm-up")
        for metric in ["nrmse", "sp_pooled", "auroc_s1"]:
            t = over_datasets(d[d.lam == 0], ["lr", "wu"], metric).unstack("wu")
            print(metric)
            print(t.round(3).to_string())
        print("\n-- per dataset nrmse, lambda 0, warm-up 0: rows lr")
        t = grid[grid.lam == 0].groupby(["lr", "dataset"])["nrmse"].mean().unstack("dataset")[ORDER]
        print(t.round(3).to_string())
        print("   worst / best over lr:", (t.max() / t.min()).round(2).to_dict())
        for lr in (1e-3, 1e-4):
            t = grid[grid.lr == lr].groupby(["lam", "dataset"])["nrmse"].mean().unstack("dataset")[ORDER]
            print("\n-- per dataset nrmse, lr %g, warm-up 0: rows lambda" % lr)
            print(t.round(3).to_string())
            print("   worst / best over lambda:", (t.max() / t.min()).round(2).to_dict())

    paper = df[(df["mode"] == "raw") & (df.lr == 1e-4) & (df.wu == 0)]
    t = paper.groupby(["dataset", "lam"])["mae_val"].mean().unstack("lam").loc[ORDER]
    print("\n== validation MAE, paper setting (raw targets, lr 1e-4), rows dataset, columns lambda")
    print("   paper table 11 (lambda 0/1/3/5): wine .554/.550/.552/.554, energy 1.82/2.71/2.38/2.06,")
    print("   yacht .053/.054/.048/.060, concrete 4.31/4.30/4.53/4.64, kin8nm .066/.064/.071/1.377,")
    print("   power 3.47/4.44/5.14/4.58")
    print(t.round(4).to_string())

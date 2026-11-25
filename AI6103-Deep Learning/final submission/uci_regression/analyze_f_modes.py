"""Experiment F, part 2: is the residual-origin score invariant to how the target is preprocessed?

Compares the original HCM (results/stepC) with rhcm (results/stepF_modes, and the
standardised runs of results/stepF) under five target preprocessings: raw, y * 0.1, y * 10,
centred, standardised. Learning rate 1e-3, seeds 0-4. Scores: u (paper eq. 2) for HCM,
|R_hat| for rhcm. Errors and interval scores are divided by the standard deviation of the
test targets (the stepC files do not contain the training split).

  python analyze_f_modes.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analyze_d import TINY, conformal_q

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]
MODES = ["raw", "scale0.1", "scale10", "center", "standardize"]
MODELS = [("hcm", "u", ["stepC"]), ("rhcm-detach", "R", ["stepF_modes", "stepF"]),
          ("rhcm-own", "R", ["stepF_modes", "stepF"])]


def one(path, kind):
    z = np.load(path)
    y_std, sc = float(z["test_s0_y"].std()), float(z["y_scale"][0])

    def score(name):
        R = np.abs(z[name + "_R"]) * sc
        return R if kind == "R" else R * np.abs(z[name + "_dnorm"] - 1.0)

    def resid(name):
        return np.abs(z[name + "_y"] - z[name + "_yhat"]).ravel()
    q = conformal_q(resid("val") / (score("val") + TINY), 0.90)
    h, e = q * (score("test_s0") + TINY), resid("test_s0")
    s0, s5 = score("test_s0"), score("test_s5")
    return dict(nrmse=float(np.sqrt((e ** 2).mean()) / y_std),
                spearman=float(spearmanr(s0, e).correlation),
                is_prop=float((2 * h + 20.0 * np.maximum(0.0, e - h)).mean() / y_std),
                auroc_s5=float(roc_auc_score(np.r_[np.zeros(len(s0)), np.ones(len(s5))], np.r_[s0, s5])))


def frame():
    rows = []
    for model, kind, exps in MODELS:
        for mode in MODES:
            for ds in ORDER:
                for seed in range(5):
                    for exp in exps:
                        p = os.path.join(HERE, "results", exp, "%s_%s_%s_lr0.001_lam0_wu0_s%d.npz"
                                         % (ds, model, mode, seed))
                        if os.path.exists(p):
                            rows.append(dict(model=model, mode=mode, dataset=ds, seed=seed, **one(p, kind)))
                            break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = frame()
    pd.set_option("display.width", 250)
    print("runs per cell:", sorted(df.groupby(["model", "mode", "dataset"]).size().unique()))
    cell = df.groupby(["model", "mode", "dataset"], sort=False).mean(numeric_only=True).drop(columns="seed")
    titles = {"nrmse": "test RMSE / std(y)", "spearman": "Spearman(score, |error|), clean test",
              "is_prop": "interval score, width proportional to the score, clean, nominal 0.90",
              "auroc_s5": "AUROC clean vs noisy inputs, sigma = 5"}
    for v, title in titles.items():
        t = cell[v].unstack("dataset")[ORDER]
        t["MEAN"] = t.mean(axis=1)
        print("\n== " + title)
        print(t.round(3).to_string())
        g = t.groupby("model", sort=False)
        print("-- range over the five preprocessings (max - min)")
        print((g.max() - g.min()).round(3).to_string())
    cell.to_csv(os.path.join(HERE, "results", "stepF_modes", "summary.csv"))

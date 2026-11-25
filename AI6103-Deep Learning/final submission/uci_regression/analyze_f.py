"""Experiment F: do our two changes to HCM fix the problems found in C, D and E?

  hcmp   HCM heads + a separate MSE prediction head (fixes the accuracy cost)
  rhcm   residual-origin HCM: the HCM heads decompose the residual of the prediction head
         (the score should then track the size of the error and be invariant to an affine
         transform of the target)

Offline analysis, no retraining. New runs are read from results/stepF, the references
(plain MLP, original HCM, Deep Ensemble, MC Dropout) from results/stepD: same datasets,
seeds 0-19, standardised targets, learning rate 1e-3.

Scores, all in the units of the original target:
  u      paper eq. (2): |R_hat| * | ||d_hat|| - 1 |
  R      |R_hat| alone (for rhcm this estimates sqrt(2) * E|residual|)
  sigma  predictive standard deviation (Deep Ensemble, MC Dropout)

Metrics per run:
  rmse, and rmse / rmse of the plain MLP with the same dataset and seed
  spearman   rank correlation of the score with |error| on the clean test set
  auroc_s1, auroc_s5   clean test inputs against inputs with Gaussian noise (sigma = 1, 5)
  split conformal intervals at nominal 0.90 (see analyze_d.py): constant width, width
  proportional to the score, and the floored rule; coverage and interval score

Combined intervals. With branch = detach or own the predictor does not depend on the HCM
heads, so `hcmp-<branch>` and `rhcm-<branch>` of one seed have the same predictions and
their heads can be read as heads of ONE network: target-origin heads (score u_t, detects
unusual inputs) and residual-origin heads (score R_r, tracks the size of the error).
Width rules, both fixed before looking at the test results, u95 and R_mean from the
training split:
  product   s(x) = R_r(x) * max(1, u_t(x) / u95)
  max       s(x) = max(R_r(x) / R_mean, u_t(x) / u95)

  python analyze_f.py          # prints the tables, writes results/stepF/summary.csv
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analyze_d import FLOOR_Q, TINY, conformal_q

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]
P = 0.90
SETS = ["test_s%d" % k for k in range(6)]
SUFFIX = "_standardize_lr0.001_lam0_wu0"
MODELS = [("mlp", "stepD", ()), ("hcm", "stepD", ("u",)),
          ("ens", "stepD", ("sigma",)), ("mcd", "stepD", ("sigma",))]
MODELS += [("%s-%s" % (m, b), "stepF", ("u", "R"))
           for m in ("hcmp", "rhcm") for b in ("shared", "detach", "own")]


def get_score(z, name, kind):
    if kind == "sigma":
        return z[name + "_sigma"].ravel()
    R = np.abs(z[name + "_R"]) * float(z["y_scale"][0])
    return R if kind == "R" else R * np.abs(z[name + "_dnorm"] - 1.0)


def interval_metrics(z, kind, y_std):
    """Split conformal intervals built from one score; `kind` None = constant width only."""
    def resid(name):
        return np.abs(z[name + "_y"] - z[name + "_yhat"]).ravel()

    e_cal = resid("val")
    rules = {"const": (lambda name: np.ones_like(resid(name)))}
    if kind is not None:
        ref = max(float(np.quantile(get_score(z, "train", kind), FLOOR_Q)), TINY)
        rules["prop"] = lambda name: get_score(z, name, kind) + TINY
        rules["floor"] = lambda name: np.maximum(1.0, get_score(z, name, kind) / ref)
    out = {}
    for rule, s_fn in rules.items():
        q = conformal_q(e_cal / s_fn("val"), P)
        for name in SETS:
            h, e = q * s_fn(name), resid(name)
            tag = "%s_%s" % (rule, name[5:])
            out["cov_" + tag] = float((e <= h).mean())
            out["is_" + tag] = float((2 * h + 2.0 / (1 - P) * np.maximum(0.0, e - h)).mean() / y_std)
        out["width_" + rule] = float((2 * q * s_fn("test_s0")).mean() / y_std)
    return out


def rows():
    for model, exp, kinds in MODELS:
        pattern = os.path.join(HERE, "results", exp, "*_%s%s_s*.npz" % (model, SUFFIX))
        for path in sorted(glob.glob(pattern)):
            base = os.path.basename(path)
            ds, seed = base.split("_")[0], int(base[:-4].split("_s")[-1])
            z = np.load(path)
            y_std = float(z["train_y"].std())
            err = np.abs(z["test_s0_y"] - z["test_s0_yhat"]).ravel()
            common = dict(model=model, dataset=ds, seed=seed,
                          rmse=float(np.sqrt((err ** 2).mean())),
                          rmse_train=float(np.sqrt(((z["train_y"] - z["train_yhat"]) ** 2).mean())))
            if "test_s0_dnorm" in z:
                common["dnorm"] = float(z["test_s0_dnorm"].mean())
            for kind in kinds or (None,):
                r = dict(common, score=kind or "-")
                r.update(interval_metrics(z, kind, y_std))
                if kind is not None:
                    s0 = get_score(z, "test_s0", kind)
                    r["spearman"] = float(spearmanr(s0, err).correlation)
                    for s in (1, 5):
                        sn = get_score(z, "test_s%d" % s, kind)
                        lab = np.r_[np.zeros(len(s0)), np.ones(len(sn))]
                        r["auroc_s%d" % s] = float(roc_auc_score(lab, np.r_[s0, sn]))
                yield r


def combined_rows():
    def load(model, ds, seed):
        return np.load(os.path.join(HERE, "results", "stepF", "%s_%s%s_s%d.npz" % (ds, model, SUFFIX, seed)))

    for branch in ("detach", "own"):
        for ds in ORDER:
            for seed in range(20):
                a, b = load("hcmp-" + branch, ds, seed), load("rhcm-" + branch, ds, seed)
                assert np.allclose(a["test_s5_yhat"], b["test_s5_yhat"], atol=1e-5)
                y_std = float(a["train_y"].std())
                u95 = max(float(np.quantile(get_score(a, "train", "u"), FLOOR_Q)), TINY)
                r_mean = max(float(get_score(b, "train", "R").mean()), TINY)

                def resid(name):
                    return np.abs(b[name + "_y"] - b[name + "_yhat"]).ravel()
                rules = {
                    "floor(u_t)": lambda n: np.maximum(1.0, get_score(a, n, "u") / u95),
                    "R_r": lambda n: get_score(b, n, "R") + TINY,
                    "product": lambda n: (get_score(b, n, "R") + TINY) * np.maximum(1.0, get_score(a, n, "u") / u95),
                    "max": lambda n: np.maximum(get_score(b, n, "R") / r_mean, get_score(a, n, "u") / u95) + TINY,
                }
                for rule, s_fn in rules.items():
                    q = conformal_q(resid("val") / s_fn("val"), P)
                    r = dict(branch=branch, dataset=ds, seed=seed, rule=rule)
                    for name in SETS:
                        h, e = q * s_fn(name), resid(name)
                        r["cov_" + name[5:]] = float((e <= h).mean())
                        r["is_" + name[5:]] = float((2 * h + 2.0 / (1 - P) * np.maximum(0.0, e - h)).mean() / y_std)
                    s0 = s_fn("test_s0")
                    r["spearman"] = float(spearmanr(s0, resid("test_s0")).correlation)
                    for k in (1, 5):
                        sn = s_fn("test_s%d" % k)
                        lab = np.r_[np.zeros(len(s0)), np.ones(len(sn))]
                        r["auroc_s%d" % k] = float(roc_auc_score(lab, np.r_[s0, sn]))
                    yield r


def frame():
    df = pd.DataFrame(list(rows()))
    ref = df[df.model == "mlp"].set_index(["dataset", "seed"])["rmse"]
    df["rmse_ratio"] = df.rmse / ref.reindex(pd.MultiIndex.from_frame(df[["dataset", "seed"]])).values
    df["name"] = df.model + np.where(df.score == "-", "", " [" + df.score + "]")
    return df


def table(df, value, fmt="%.3f"):
    t = df.groupby(["name", "dataset"], sort=False)[value].mean().unstack("dataset")[ORDER]
    t["MEAN"] = t.mean(axis=1)
    return t.applymap(lambda v: fmt % v)


if __name__ == "__main__":
    df = frame()
    df.groupby(["name", "dataset"], sort=False).mean(numeric_only=True).drop(columns="seed") \
      .to_csv(os.path.join(HERE, "results", "stepF", "summary.csv"))
    pd.set_option("display.width", 250)
    print("runs per model:", df.groupby("model", sort=False).size().div(
        df.groupby("model", sort=False).score.nunique()).astype(int).to_dict())
    one = df.drop_duplicates(["model", "dataset", "seed"])
    print("\n== test RMSE / test RMSE of the plain MLP (same seed); 1 = as accurate as the MLP")
    print(table(one.assign(name=one.model), "rmse_ratio").to_string())
    print("\n== mean ||d_hat|| on the clean test set")
    print(table(one[one.dnorm.notna()].assign(name=one.model), "dnorm").to_string())
    scored = df[df.score != "-"]
    for value, title in [("spearman", "Spearman(score, |error|), clean test"),
                         ("auroc_s1", "AUROC clean vs noisy inputs, sigma = 1"),
                         ("auroc_s5", "AUROC clean vs noisy inputs, sigma = 5")]:
        print("\n== " + title)
        print(table(scored, value).to_string())
    cols = ["cov_const_s0", "is_const_s0", "cov_prop_s0", "width_prop", "is_prop_s0", "is_prop_s1",
            "is_prop_s5", "cov_floor_s0", "is_floor_s0", "is_floor_s1", "is_floor_s5"]
    print("\n== conformal intervals, nominal %.2f, mean of six datasets" % P)
    print("   const = constant width, prop = width proportional to the score, floor = floored rule")
    per_ds = scored.groupby(["name", "dataset"], sort=False)[cols].mean()
    print(per_ds.groupby("name", sort=False).mean().round(3).to_string())
    print("\n== interval score, width proportional to the score, clean test (lower is better)")
    print(table(scored, "is_prop_s0").to_string())

    comb = pd.DataFrame(list(combined_rows()))
    comb.to_csv(os.path.join(HERE, "results", "stepF", "combined_runs.csv"), index=False)
    print("\n== one network with both pairs of heads: conformal intervals, nominal %.2f, mean of six" % P)
    per_ds = comb.groupby(["branch", "rule", "dataset"], sort=False).mean(numeric_only=True).drop(columns="seed")
    print(per_ds.groupby(["branch", "rule"], sort=False).mean().round(3).to_string())

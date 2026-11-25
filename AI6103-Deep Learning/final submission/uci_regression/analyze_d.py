"""Experiment D: turning uncertainty scores into prediction intervals.

Offline analysis of results/stepD/*.npz (no retraining). The validation split (10%) is the
calibration set; the training split is only used to fit the width model of `*_cp_fit`.

Interval = y_hat +/- half_width(x). Methods, for a nominal coverage level p:

  hcm_temp      paper, section 3.3: u_cal = T * u with one T fitted on the calibration set
                so that coverage at 1, 2, 3 x u_cal is closest to 68 / 95 / 99.7 %;
                half_width = z_p * T * u
  *_cp_const    split conformal, constant width: half_width = q,
                q = conformal quantile of |y - y_hat| on the calibration set
  *_cp_u / *_cp_sigma
                normalised split conformal: score = |y - y_hat| / s(x), half_width = q * s(x),
                with s = u (HCM) or s = predictive std (Deep Ensemble, MC Dropout)
  *_cp_fit      the same with s(x) = a + b * score(x), a, b >= 0 fitted by non-negative least
                squares of |residual| on the TRAINING split (never on calibration or test)
  *_cp_floor    the same with s(x) = max(1, score(x) / ref), ref = the 95th percentile of the
                score on the TRAINING split: constant width for inputs whose score is in the
                usual training range, wider only when the score is unusually large
  ens_gauss / mcd_gauss
                no calibration: half_width = z_p * predictive std

Split conformal gives marginal coverage >= p when calibration and test points are
exchangeable. Noisy test inputs (sigma > 0) break that, so coverage there is not guaranteed.

Metrics (mean over seeds): coverage; mean width / std(y_train); interval (Winkler) score
/ std(y_train), lower is better: width + (2 / (1 - p)) * distance of y outside the interval.

  python analyze_d.py          # prints the p = 0.90 tables, writes results/stepD/summary.csv
"""
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "results", "stepD")
ORDER = ["wine", "concrete", "energy", "kin8nm", "power", "yacht"]
LEVELS = [(0.6827, 1.0), (0.90, 1.6449), (0.9545, 2.0)]       # (coverage p, normal z_p)
SETS = ["test_s0", "test_s1", "test_s2", "test_s5"]
TINY = 1e-12
FLOOR_Q = 0.95
MODELS = {"hcm": "hcm_standardize", "hcmraw": "hcm_raw", "ens": "ens_standardize",
          "mcd": "mcd_standardize"}


def conformal_q(scores, p):
    n = len(scores)
    k = math.ceil((n + 1) * p)
    return np.inf if k > n else np.sort(scores)[k - 1]


def fit_temperature(e, u):
    """Paper's calibration: one scalar T for coverage at 1, 2, 3 sigma."""
    ratio = e / np.maximum(u, TINY)
    cand = np.unique(np.concatenate([ratio / k for k in (1, 2, 3)]))
    cand = cand[np.isfinite(cand) & (cand > 0)]
    loss = sum((np.mean(ratio[None, :] <= k * cand[:, None], axis=1) - t) ** 2
               for k, t in zip((1, 2, 3), (0.6827, 0.9545, 0.9973)))
    return cand[np.argmin(loss)]


def fit_width(score_train, e_train):
    """s(x) = a + b * score(x), a, b >= 0, least squares on the training residuals."""
    A = np.c_[np.ones_like(score_train), score_train]
    (a, b), _ = nnls(A, e_train)
    return (a, b) if a + b > 0 else (1.0, 0.0)


def half_widths(z, kind):
    """dict method -> {set -> {p -> half-width array}} for one run."""
    def resid(name):
        return np.abs(z[name + "_y"] - z[name + "_yhat"]).ravel()

    if kind in ("hcm", "hcmraw"):
        def score(name):          # u in original target units
            return (np.abs(z[name + "_R"]) * np.abs(z[name + "_dnorm"] - 1.0)) * float(z["y_scale"][0])
        tag, prefix = "u", "hcm" if kind == "hcm" else "hcmraw"
    else:
        def score(name):
            return z[name + "_sigma"].ravel()
        tag, prefix = "sigma", kind

    e_cal, s_cal = resid("val"), score("val")
    a, b = fit_width(score("train"), resid("train"))
    ref = max(float(np.quantile(score("train"), FLOOR_Q)), TINY)

    def floored(v):
        return np.maximum(1.0, v / ref)
    out = {}
    for p, zp in LEVELS:
        q_const = conformal_q(e_cal, p)
        q_norm = conformal_q(e_cal / (s_cal + TINY), p)
        q_fit = conformal_q(e_cal / (a + b * s_cal), p)
        q_floor = conformal_q(e_cal / floored(s_cal), p)
        T = fit_temperature(e_cal, s_cal) if tag == "u" else None
        for name in SETS:
            s = score(name)
            hw = {prefix + "_cp_const": np.full_like(s, q_const),
                  prefix + "_cp_" + tag: q_norm * (s + TINY),
                  prefix + "_cp_fit": q_fit * (a + b * s),
                  prefix + "_cp_floor": q_floor * floored(s)}
            if tag == "u":
                hw[prefix + "_temp"] = zp * T * s
            else:
                hw[prefix + "_gauss"] = zp * s
            for method, h in hw.items():
                out.setdefault(method, {}).setdefault(name, {})[p] = h
    return out, resid


def rows():
    for kind, tag in MODELS.items():
        for path in sorted(glob.glob(os.path.join(EXP, "*_%s_lr0.001_lam0_wu0_s*.npz" % tag))):
            base = os.path.basename(path)
            ds, seed = base.split("_")[0], int(base[:-4].split("_s")[-1])
            z = np.load(path)
            y_std = float(z["train_y"].std())
            hw_all, resid = half_widths(z, kind)
            for method, per_set in hw_all.items():
                for name, per_p in per_set.items():
                    e = resid(name)
                    for p, h in per_p.items():
                        miss = np.maximum(0.0, e - h)
                        yield dict(dataset=ds, seed=seed, method=method, set=name, p=p,
                                   cov=float((e <= h).mean()),
                                   width=float((2 * h).mean() / y_std),
                                   width_med=float(np.median(2 * h) / y_std),
                                   iscore=float((2 * h + 2.0 / (1 - p) * miss).mean() / y_std))


def table(df, value, p, name, fmt="%.3f"):
    sub = df[(df.p == p) & (df.set == name)]
    t = sub.groupby(["method", "dataset"])[value].mean().unstack("dataset")[ORDER]
    t["MEAN"] = t.mean(axis=1)
    return t.applymap(lambda v: fmt % v)


if __name__ == "__main__":
    df = pd.DataFrame(list(rows()))
    df.groupby(["method", "dataset", "set", "p"])[["cov", "width", "width_med", "iscore"]] \
      .agg(["mean", "std"]).to_csv(os.path.join(EXP, "summary.csv"))
    pd.set_option("display.width", 250)
    print("seeds per cell:", sorted(df.groupby(["method", "dataset", "set", "p"]).size().unique()))
    for p in (0.90,):
        print("\n== coverage, clean test, nominal %.2f" % p)
        print(table(df, "cov", p, "test_s0").to_string())
        print("\n== mean width / std(y), clean test, nominal %.2f" % p)
        print(table(df, "width", p, "test_s0").to_string())
        print("\n== interval score / std(y), clean test, nominal %.2f (lower is better)" % p)
        print(table(df, "iscore", p, "test_s0").to_string())
        for name in ("test_s1", "test_s5"):
            print("\n== coverage under input noise (%s), nominal %.2f" % (name, p))
            print(table(df, "cov", p, name).to_string())
        print("\n== interval score under input noise (test_s1), nominal %.2f" % p)
        print(table(df, "iscore", p, "test_s1").to_string())
    print("\n== clean coverage at the other levels (mean over datasets)")
    other = df[df.set == "test_s0"].groupby(["method", "p"])["cov"].mean().unstack("p")
    print(other.round(3).to_string())

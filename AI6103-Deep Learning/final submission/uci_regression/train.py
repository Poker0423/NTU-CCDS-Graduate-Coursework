"""Train one model on one UCI dataset and dump per-sample outputs.

One run = (dataset, seed, method, hyper-parameters). Everything needed for later analysis
(target scaling, conformal calibration, sensitivity) is saved to an .npz file, so those
analyses never need retraining.

Methods:
  mlp   plain regression, MSE
  hcm   HCM direction + magnitude heads (models.hcm_loss)
  hcmp  ours: HCM heads + a separate MSE prediction head; the prediction is no longer
        R_hat * d_hat, the score is still u computed from the HCM heads
  rhcm  ours, residual-origin HCM: the HCM heads decompose the residual
        r = (t - f(x)) / s of the prediction head f, with s the running RMS of the
        residual. The origin of the decomposition is the model's own prediction instead
        of the arbitrary zero of the target space.
        (`--branch` shared | detach | own: how the HCM heads use the trunk, see models.HCMPlus)
  ens   Deep Ensemble: `members` Gaussian-NLL networks with different initialisations
  mcd   MC Dropout: MSE with dropout `dropout`, `mc_passes` stochastic passes at test time

Settings taken from the HCM paper (appendix A.1, UCI): MLP 3x20 LeakyReLU(0.01), MSE,
Adam lr 1e-4, betas (0.9, 0.999), weight decay 1e-4, 200 epochs, 80/10/10 split,
test-time input noise N(0, sigma^2 I) with sigma in {0,...,5}.

Choices the paper does not state (ours, keep them fixed across the group):
  - batch size 64
  - inputs standardised with training-set statistics; test noise is added after that
  - the weights of the last epoch are evaluated (no early stopping)
  - target preprocessing is a flag (--target_mode); `raw` is the default
  - ens: 5 members; mcd: dropout 0.1, 50 passes
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import data
from models import MLP, GaussMLP, HCMNet, HCMPlus, expand_target, gauss_nll, hcm_loss, hcm_outputs

SIGMAS = [0, 1, 2, 3, 4, 5]
HCM_LIKE = ("hcm", "hcmp", "rhcm")      # methods that output R_hat and ||d_hat||
HERE = os.path.dirname(os.path.abspath(__file__))


def target_transform(mode, y_train):
    """Return (shift, scale) so that t = (y - shift) / scale."""
    k = y_train.shape[1]
    if mode == "raw":
        return np.zeros(k), np.ones(k)
    if mode == "standardize":
        return y_train.mean(0), y_train.std(0)
    if mode == "center":
        return y_train.mean(0), np.ones(k)
    if mode.startswith("scale:"):          # t = c * y
        return np.zeros(k), np.ones(k) / float(mode.split(":")[1])
    raise ValueError(mode)


def per_sample_error(y, y_hat):
    """Root mean squared error over the target dimensions (absolute error when k = 1)."""
    return np.sqrt(((y - y_hat) ** 2).mean(axis=1))


def fit(model, loss_fn, Xtr, Ttr, cfg):
    """Mini-batch Adam; returns the loss of the last batch (and its parts, if any)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                           weight_decay=cfg["wd"])
    n, bs, last = len(Xtr), cfg["batch"], (float("nan"),)
    for epoch in range(cfg["epochs"]):
        model.train()
        # Warm-up: 0.1 x learning rate for the first `warmup_epochs` epochs. The official
        # CIFAR script multiplies the LOSS by 0.1 for its first epochs; with SGD that is the
        # same thing. With Adam a loss factor is cancelled by the update rule, so the
        # learning rate is scaled instead.
        for group in opt.param_groups:
            group["lr"] = cfg["lr"] * (0.1 if epoch < cfg["warmup_epochs"] else 1.0)
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, parts = loss_fn(model, Xtr[idx], Ttr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = (loss.item(),) + tuple(parts)
        if not np.isfinite(last[0]):
            break
    model.eval()
    return last


def run(cfg):
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    t0 = time.time()

    X, Y = data.load(cfg["dataset"])
    tr, va, te = data.split(len(X), cfg["seed"])
    x_mu, x_sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    shift, scale = target_transform(cfg["target_mode"], Y[tr])

    def fx(a):
        return torch.tensor((a - x_mu) / x_sd, dtype=torch.float32)

    def fy(a):
        return torch.tensor((a - shift) / scale, dtype=torch.float32)

    Xtr, Ttr = fx(X[tr]), fy(Y[tr])
    p, k = X.shape[1], Y.shape[1]
    method = cfg["method"]

    # ---------- training ----------
    if method == "hcm":
        model = HCMNet(p, expand_target(Ttr).shape[1])

        def loss_fn(mod, x, t):
            return hcm_loss(*mod(x), expand_target(t), cfg["lam"])
        last = fit(model, loss_fn, Xtr, Ttr, cfg)
    elif method in ("hcmp", "rhcm"):
        model = HCMPlus(p, k, expand_target(Ttr).shape[1], cfg["branch"])

        def loss_fn(mod, x, t):
            y_hat, d_hat, R_hat = mod(x)
            mse = ((y_hat - t) ** 2).mean()
            if method == "hcmp":            # HCM on the target, as in the paper
                target = t
            else:                           # HCM on the standardised residual
                r = (t - y_hat).detach()
                target = r / mod.residual_scale(r)
            h_loss, parts = hcm_loss(d_hat, R_hat, expand_target(target), cfg["lam"])
            return mse + h_loss, (mse.item(),) + parts
        last = fit(model, loss_fn, Xtr, Ttr, cfg)
    elif method == "mlp" or method == "mcd":
        model = MLP(p, k, dropout=cfg["dropout"] if method == "mcd" else 0.0)
        last = fit(model, lambda mod, x, t: (((mod(x) - t) ** 2).mean(), ()), Xtr, Ttr, cfg)
    elif method == "ens":
        members = []
        for j in range(cfg["members"]):
            torch.manual_seed(1000 * (cfg["seed"] + 1) + j)
            member = GaussMLP(p, k)
            last = fit(member, lambda mod, x, t: (gauss_nll(*mod(x), t), ()), Xtr, Ttr, cfg)
            members.append(member)
    else:
        raise ValueError(method)

    def predict(x):
        """Prediction in transformed target space, plus method-specific extras."""
        if method == "hcm":
            t_hat, R_hat, d_norm = hcm_outputs(*model(x), k)
            return t_hat, {"R": R_hat.numpy(), "dnorm": d_norm.numpy()}
        if method in ("hcmp", "rhcm"):
            y_hat, d_hat, R_hat = model(x)
            # rhcm: R_hat is in units of the residual scale s; store it in target units.
            # `rhat` = s * R_hat * d_hat is the HCM heads' own estimate of the residual.
            s_res = model.residual_scale().item() if method == "rhcm" else 1.0
            _, R1, d_norm = hcm_outputs(d_hat, R_hat, k)
            extra = {"R": R1.numpy() * s_res, "dnorm": d_norm.numpy()}
            if method == "rhcm":
                extra["rhat"] = hcm_outputs(d_hat, R_hat, k)[0].numpy() * s_res * scale
            return y_hat, extra
        if method == "mlp":
            return model(x), {}
        if method == "mcd":           # dropout stays on: mean and std of the passes
            model.train()
            torch.manual_seed(cfg["seed"])
            passes = torch.stack([model(x) for _ in range(cfg["mc_passes"])])
            model.eval()
            return passes.mean(0), {"sigma": passes.std(0).numpy() * scale}
        # Deep Ensemble: mixture of the members' Gaussians
        mus = torch.stack([mem(x)[0] for mem in members])
        var = torch.stack([mem(x)[1].exp() for mem in members])
        mean = mus.mean(0)
        total = (var + mus ** 2).mean(0) - mean ** 2
        return mean, {"sigma": total.clamp_min(0).sqrt().numpy() * scale}

    # ---------- evaluation ----------
    noise_rng = np.random.RandomState(10_000 + cfg["seed"])
    sets = {"train": (Xtr, Y[tr]), "val": (fx(X[va]), Y[va])}
    x_te = fx(X[te])
    for s in SIGMAS:
        eps = torch.tensor(noise_rng.randn(*x_te.shape), dtype=torch.float32)
        sets["test_s%d" % s] = (x_te + s * eps, Y[te])

    out, m = {}, dict(cfg)
    with torch.no_grad():
        for name, (x, y) in sets.items():
            t_hat, extra = predict(x)
            y_hat = t_hat.numpy() * scale + shift
            out[name + "_y"], out[name + "_yhat"] = y, y_hat
            for key, val in extra.items():
                out[name + "_" + key] = val
            m["rmse_" + name] = float(np.sqrt(((y - y_hat) ** 2).mean()))

    m["diverged"] = not all(np.isfinite(v).all() for v in out.values())
    if method in HCM_LIKE and not m["diverged"]:
        # u = |R_hat| * | ||d_hat|| - 1 |  (paper eq. 2), in the units of the transformed target
        u = {nm: np.abs(out[nm + "_R"]) * np.abs(out[nm + "_dnorm"] - 1.0) for nm in sets}
        err = {nm: per_sample_error(out[nm + "_y"], out[nm + "_yhat"]) for nm in sets}
        for s in SIGMAS:
            nm = "test_s%d" % s
            m["u_mean_s%d" % s] = float(u[nm].mean())
            m["dnorm_mean_s%d" % s] = float(out[nm + "_dnorm"].mean())
            if s > 0:  # can u tell clean test inputs from noisy ones?
                lab = np.r_[np.zeros(len(u["test_s0"])), np.ones(len(u[nm]))]
                m["auroc_s%d" % s] = float(roc_auc_score(lab, np.r_[u["test_s0"], u[nm]]))
        m["spearman_clean"] = float(spearmanr(u["test_s0"], err["test_s0"]).correlation)
        tests = ["test_s%d" % s for s in SIGMAS]
        m["spearman_pooled"] = float(spearmanr(np.concatenate([u[t] for t in tests]),
                                               np.concatenate([err[t] for t in tests])).correlation)
        m["u_val_mean"] = float(u["val"].mean())
        # the magnitude head alone as a score (the natural score for rhcm)
        aR = {nm: np.abs(out[nm + "_R"]) for nm in sets}
        m["spearman_clean_R"] = float(spearmanr(aR["test_s0"], err["test_s0"]).correlation)
        for s in SIGMAS[1:]:
            nm = "test_s%d" % s
            lab = np.r_[np.zeros(len(aR["test_s0"])), np.ones(len(aR[nm]))]
            m["auroc_R_s%d" % s] = float(roc_auc_score(lab, np.r_[aR["test_s0"], aR[nm]]))

    m.update(final_loss=last[0], loss_parts=list(last[1:]), seconds=round(time.time() - t0, 1),
             n_train=int(len(Xtr)), y_shift=shift.tolist(), y_scale=scale.tolist())

    out_dir = os.path.join(HERE, "results", cfg["exp"])
    os.makedirs(out_dir, exist_ok=True)
    tag = "%s_%s_s%d" % (cfg["dataset"], cfg["tag"], cfg["seed"])
    np.savez_compressed(os.path.join(out_dir, tag + ".npz"), y_shift=shift, y_scale=scale, **out)
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def make_cfg(**kw):
    cfg = dict(exp="debug", dataset="kin8nm", seed=0, method="hcm", lr=1e-4, wd=1e-4,
               epochs=200, batch=64, lam=0.0, warmup_epochs=0, target_mode="raw",
               members=5, dropout=0.1, mc_passes=50, branch="shared")
    cfg.update(kw)
    if "tag" not in cfg:
        name = cfg["method"] + ("-" + cfg["branch"] if cfg["method"] in ("hcmp", "rhcm") else "")
        cfg["tag"] = "%s_%s_lr%g_lam%g_wu%d" % (
            name, cfg["target_mode"].replace(":", ""), cfg["lr"], cfg["lam"],
            cfg["warmup_epochs"])
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="debug")
    ap.add_argument("--dataset", default="kin8nm", choices=data.ALL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="hcm", choices=["mlp", "hcm", "ens", "mcd", "hcmp", "rhcm"])
    ap.add_argument("--branch", default="shared", choices=["shared", "detach", "own"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--target_mode", default="raw")
    res = run(make_cfg(**vars(ap.parse_args())))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))

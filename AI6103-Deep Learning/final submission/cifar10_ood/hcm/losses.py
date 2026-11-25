"""Losses for CE and HCM training.

All losses take *soft* targets y of shape (B, C) (one-hot for plain training, mixed
labels under MixUp), so the same code path serves both settings.

HCM target decomposition (y = R* . d*)
--------------------------------------
For one-hot labels every choice gives R* = 1, d* = y. Under MixUp the paper does not say
how the soft label is decomposed, so two options are implemented (`mix_target`):

  "soft"   (default) d* = y (norm < 1 for mixed labels), R* = 1.
           This is what you get by feeding the soft label into the official CIFAR-10 loss
           unchanged, and it matches the paper's explanation of HCM-mix (Sec. 3.2: mixed
           samples have directions "between class anchors" whose magnitude is below one,
           so ||d|| deviates from 1 and u(x) grows).
  "decomp" d* = y / ||y||, R* = ||y||  (the literal hyperspherical decomposition of y).
           d* is always unit-norm, so MixUp then only teaches R.

HCM loss forms (`hcm_loss`)
---------------------------
  "official" (default) - exactly HCM_CIFAR10.py for one-hot targets:
        L = MSE(d, d*)^p + MSE(R d*, R* d*)^p              with p = 3/4
        (+ lambda_norm * MSE(||d||, 1)^p if lambda_norm > 0)
  "paper"   - paper Eq. 3 with phi(t) = mean |t|^p (p = 2 -> squared loss):
        L = phi(R ||d - d*||) + phi(R - R*) + lambda_norm * phi(||d|| - 1)
"""
import torch
import torch.nn.functional as F


def soft_cross_entropy(logits, y_soft):
    """CE with soft targets; equals F.cross_entropy for one-hot targets."""
    return -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def hcm_targets(y_soft, mix_target="soft", eps=1e-12):
    if mix_target == "soft":
        d_t = y_soft
        R_t = torch.ones(y_soft.size(0), device=y_soft.device, dtype=y_soft.dtype)
    elif mix_target == "decomp":
        R_t = y_soft.norm(dim=1).clamp_min(eps)
        d_t = y_soft / R_t.unsqueeze(1)
    else:
        raise ValueError(mix_target)
    return d_t, R_t


def _pow(x, p, eps=1e-12):
    # clamp avoids an infinite gradient of x**p at x == 0 when p < 1
    return x.clamp_min(eps) ** p


def hcm_loss(d, R, y_soft, form="official", mix_target="soft", power=None, lambda_norm=0.0):
    """Returns (total_loss, dict_of_components_as_floats)."""
    d_t, R_t = hcm_targets(y_soft, mix_target)
    dn = d.norm(dim=1)
    if form == "official":
        p = 0.75 if power is None else power
        l_d = F.mse_loss(d, d_t)
        l_r = F.mse_loss(R.unsqueeze(1) * d_t, R_t.unsqueeze(1) * d_t)
        loss = _pow(l_d, p) + _pow(l_r, p)
        l_n = F.mse_loss(dn, torch.ones_like(dn))
        if lambda_norm > 0:
            loss = loss + lambda_norm * _pow(l_n, p)
    elif form == "paper":
        p = 2.0 if power is None else power
        phi = lambda t: t.abs().pow(p).mean()
        l_d = phi(R * (d - d_t).norm(dim=1))
        l_r = phi(R - R_t)
        l_n = phi(dn - 1.0)
        loss = l_d + l_r + (lambda_norm * l_n if lambda_norm > 0 else 0.0)
    else:
        raise ValueError(form)
    comps = {"loss_d": l_d.item(), "loss_r": l_r.item(), "loss_norm": l_n.item()}
    return loss, comps

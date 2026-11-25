"""Models and the HCM loss for tabular regression.

HCM (Choi et al., ICLR 2026) writes a target y in R^D as y = R * d with R = ||y||_2 and
||d||_2 = 1. The network predicts d_hat (direction head) and R_hat (magnitude head); the
prediction is R_hat * d_hat and the uncertainty score is u = R_hat * | ||d_hat||_2 - 1 |.
A scalar target is embedded as (y, y), following section 2.2 of the paper.
"""
import torch
import torch.nn as nn

WIDTH, DEPTH = 20, 3  # paper, appendix A.1 (UCI): three hidden layers of 20 units


def _trunk(p, dropout=0.0):
    layers, d = [], p
    for _ in range(DEPTH):
        layers += [nn.Linear(d, WIDTH), nn.LeakyReLU(0.01)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = WIDTH
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Plain regression baseline: same trunk, one linear output head.

    With dropout > 0 it is the MC Dropout baseline (Gal & Ghahramani, 2016): keep dropout
    active at test time and use the spread of several stochastic passes.
    """

    def __init__(self, p, k, dropout=0.0):
        super().__init__()
        self.trunk = _trunk(p, dropout)
        self.out = nn.Linear(WIDTH, k)

    def forward(self, x):
        return self.out(self.trunk(x))


class GaussMLP(nn.Module):
    """One Deep Ensemble member (Lakshminarayanan et al., 2017): predicts a mean and a
    log-variance per target and is trained with the Gaussian negative log-likelihood."""

    def __init__(self, p, k):
        super().__init__()
        self.trunk = _trunk(p)
        self.mean = nn.Linear(WIDTH, k)
        self.logvar = nn.Linear(WIDTH, k)

    def forward(self, x):
        h = self.trunk(x)
        return self.mean(h), self.logvar(h).clamp(-10.0, 10.0)


def gauss_nll(mean, logvar, t):
    return 0.5 * (logvar + (t - mean) ** 2 / logvar.exp()).mean()


class HCMNet(nn.Module):
    """Same trunk, with a direction head (D outputs) and a magnitude head (1 output)."""

    def __init__(self, p, D):
        super().__init__()
        self.trunk = _trunk(p)
        self.head_d = nn.Linear(WIDTH, D)
        self.head_R = nn.Linear(WIDTH, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_d(h), self.head_R(h)


class HCMPlus(nn.Module):
    """HCM heads plus a separate prediction head (our variants `hcmp` and `rhcm`).

    The prediction comes from `head_y`, trained with the plain MSE; the direction and
    magnitude heads only produce the uncertainty score.

    branch = "shared": the three heads read one trunk and every loss term trains it
             "detach": the HCM heads read the trunk features but send no gradient back, so
                       the predictor is trained exactly like the plain MLP
             "own":    the HCM heads have their own trunk (about twice the parameters)
    """

    def __init__(self, p, k, D, branch="shared"):
        super().__init__()
        assert branch in ("shared", "detach", "own")
        self.branch = branch
        self.trunk = _trunk(p)
        self.head_y = nn.Linear(WIDTH, k)
        self.utrunk = _trunk(p) if branch == "own" else None
        self.head_d = nn.Linear(WIDTH, D)
        self.head_R = nn.Linear(WIDTH, 1)
        # running mean of the squared residual and its number of updates (used by `rhcm`)
        self.register_buffer("res_ms", torch.zeros(()))
        self.register_buffer("res_n", torch.zeros(()))

    def forward(self, x):
        h = self.trunk(x)
        y_hat = self.head_y(h)
        if self.branch == "own":
            g = self.utrunk(x)
        else:
            g = h.detach() if self.branch == "detach" else h
        return y_hat, self.head_d(g), self.head_R(g)

    def residual_scale(self, r=None, beta=0.99):
        """RMS of the residual: bias-corrected exponential moving average over the training
        batches. Pass the current residuals `r` to update it (training only)."""
        if r is not None:
            self.res_ms.mul_(beta).add_((1 - beta) * (r ** 2).mean())
            self.res_n.add_(1)
        if self.res_n.item() == 0:
            return torch.ones(())
        return (self.res_ms / (1 - beta ** self.res_n.item())).sqrt().clamp_min(1e-8)


def expand_target(t):
    """(n, k) -> (n, D). A scalar target is duplicated to (t, t); k >= 2 is kept as is."""
    return t.repeat(1, 2) if t.shape[1] == 1 else t


def decompose(y_exp, eps=1e-8):
    """Ground-truth magnitude R (n, 1) and unit direction d (n, D)."""
    R = y_exp.norm(dim=1, keepdim=True)
    return R, y_exp / R.clamp_min(eps)


def hcm_loss(d_hat, R_hat, y_exp, lam=0.0):
    """Paper eq. (3) with the squared loss for every term.

    loss_R = MSE(R_hat * d, y)   -> magnitude error e_R^2 (because ||d|| = 1)
    loss_d = MSE(R * d_hat, y)   -> direction error weighted by the true magnitude, R^2 ||e_d||^2
    norm   = mean((||d_hat|| - 1)^2), weighted by lam (lam = 0 reproduces the official scripts)
    The first two terms are exactly what the official Toy_example.py computes.
    """
    R, d = decompose(y_exp)
    loss_R = ((R_hat * d - y_exp) ** 2).mean()
    loss_d = ((R * d_hat - y_exp) ** 2).mean()
    norm_pen = ((d_hat.norm(dim=1) - 1.0) ** 2).mean()
    return loss_R + loss_d + lam * norm_pen, (loss_R.item(), loss_d.item(), norm_pen.item())


def hcm_outputs(d_hat, R_hat, k):
    """Prediction in target space (n, k), plus R_hat (n,) and ||d_hat|| (n,)."""
    y_exp_hat = R_hat * d_hat
    y_hat = y_exp_hat.mean(dim=1, keepdim=True) if k == 1 else y_exp_hat
    return y_hat, R_hat.squeeze(1), d_hat.norm(dim=1)

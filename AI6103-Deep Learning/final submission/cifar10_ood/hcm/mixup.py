"""MixUp variants mentioned in the paper (Appendix A.2):
  k = 2   : classic two-sample MixUp, lambda ~ Beta(alpha, alpha)
  k > 2   : Dirichlet multi-sample MixUp, w ~ Dir(alpha * 1_k)   (paper: k = 20, alpha = 0.5)

Partners are random permutations of the current batch (the first partner is the batch
itself). By default one weight vector is drawn per batch (as in Zhang et al., 2018);
`per_sample=True` draws one per example.
"""
import torch


def mix_batch(x, y_soft, k=2, alpha=1.0, per_sample=False, prob=1.0):
    """x: (B, ...) images, y_soft: (B, C) one-hot labels. Returns mixed (x, y)."""
    if k < 2 or alpha <= 0 or torch.rand(1).item() >= prob:
        return x, y_soft
    B = x.size(0)
    n_draw = B if per_sample else 1
    conc = torch.full((k,), float(alpha))
    if k == 2:
        lam = torch.distributions.Beta(conc[0], conc[1]).sample((n_draw,))
        w = torch.stack([lam, 1 - lam], dim=1)
    else:
        w = torch.distributions.Dirichlet(conc).sample((n_draw,))
    w = w.to(x.device, x.dtype)                         # (n_draw, k)
    perms = [torch.arange(B, device=x.device)] + [torch.randperm(B, device=x.device) for _ in range(k - 1)]
    view = (-1,) + (1,) * (x.dim() - 1)
    x_mix = sum(w[:, j].view(view) * x[perms[j]] for j in range(k))
    y_mix = sum(w[:, j].view(-1, 1) * y_soft[perms[j]] for j in range(k))
    return x_mix, y_mix

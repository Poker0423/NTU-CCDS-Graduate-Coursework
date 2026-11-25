"""HCM uncertainty scores. Convention: HIGHER = MORE UNCERTAIN (more likely OOD).

All functions take numpy arrays or torch tensors: d (N, C), R (N,).

  paper_eq2      u = R * | ||d|| - 1 |                    (paper Eq. 2, README)
  official_cifar u = 1 - c,  c = exp(-|n-1|) if n >= 1 else exp(-|1/n - 1|),  n = ||d||
                 (HCM_CIFAR10.py `compute_final_confidence`; no R)
  official_toy   u = R * sqrt(| ||d||^2 - 1 |)            (Toy_example.py)
  official_moons u = | ||R d|| - 1 |                      (two_moons.py)
  norm_dev       u = | ||d|| - 1 |                        (README's classification form)
"""
import numpy as np

try:
    import torch
except ImportError:            # scores can be computed from .npz files without torch
    torch = None


def _np(a):
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().float().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def paper_eq2(d, R):
    d, R = _np(d), _np(R)
    return R * np.abs(np.linalg.norm(d, axis=1) - 1.0)


def norm_dev(d, R=None):
    return np.abs(np.linalg.norm(_np(d), axis=1) - 1.0)


def official_cifar(d, R=None):
    n = np.clip(np.linalg.norm(_np(d), axis=1), 1e-12, None)
    conf = np.where(n >= 1.0, np.exp(-np.abs(n - 1.0)), np.exp(-np.abs(1.0 / n - 1.0)))
    return 1.0 - conf


def official_toy(d, R):
    d, R = _np(d), _np(R)
    return R * np.sqrt(np.abs((d ** 2).sum(1) - 1.0))


def official_moons(d, R):
    d, R = _np(d), _np(R)
    return np.abs(np.linalg.norm(R[:, None] * d, axis=1) - 1.0)


SCORES = {
    "paper_eq2": paper_eq2,
    "official_cifar": official_cifar,
    "official_toy": official_toy,
    "official_moons": official_moons,
    "norm_dev": norm_dev,
}

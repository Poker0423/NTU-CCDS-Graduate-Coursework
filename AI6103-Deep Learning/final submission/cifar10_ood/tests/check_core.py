"""Quick correctness checks (CPU, ~10 s). Run from the package root:
    python tests/check_core.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hcm.losses import hcm_loss, soft_cross_entropy  # noqa: E402
from hcm.mixup import mix_batch  # noqa: E402
from hcm.models import build_model  # noqa: E402

torch.manual_seed(0)
B, C = 16, 10
y = torch.randint(0, C, (B,))
y1 = F.one_hot(y, C).float()
d, R = torch.randn(B, C), torch.randn(B)

# 1) our "official" HCM loss == HCM_CIFAR10.py train_one_epoch for one-hot labels
mse = nn.MSELoss()
ref = mse(d, y1) ** 0.75 + mse(R.view(-1, 1) * y1, y1) ** 0.75
ours, _ = hcm_loss(d, R, y1, "official")
assert torch.allclose(ours, ref, atol=1e-6), (ours, ref)
# both target decompositions coincide on one-hot labels
assert torch.allclose(hcm_loss(d, R, y1, "official", "decomp")[0], ref, atol=1e-6)
print("[ok] official HCM loss matches the official script")

# 2) soft CE == F.cross_entropy on one-hot targets
logits = torch.randn(B, C)
assert torch.allclose(soft_cross_entropy(logits, y1), F.cross_entropy(logits, y), atol=1e-6)
print("[ok] soft-target CE")

# 3) MixUp labels stay on the simplex; images are the same convex combination
x = torch.randn(B, 3, 32, 32)
for k, alpha, ps in [(2, 1.0, False), (2, 1.0, True), (20, 0.5, False), (20, 0.5, True)]:
    xm, ym = mix_batch(x, y1, k, alpha, ps)
    assert xm.shape == x.shape and ym.shape == y1.shape
    assert torch.allclose(ym.sum(1), torch.ones(B), atol=1e-5) and (ym >= 0).all()
print("[ok] MixUp k=2 / Dirichlet k=20")

# 4) paper loss runs and is differentiable
dd, RR = d.clone().requires_grad_(), R.clone().requires_grad_()
l, comps = hcm_loss(dd, RR, ym, "paper", "soft", None, lambda_norm=1.0)
l.backward()
assert torch.isfinite(dd.grad).all() and torch.isfinite(RR.grad).all()
print("[ok] paper Eq.3 loss + gradients", {k: round(v, 4) for k, v in comps.items()})

# 5) model outputs + official checkpoint key names
for m in ("ce", "hcm"):
    net = build_model(m).eval()
    out = net(torch.randn(2, 3, 32, 32))
    assert out["logits"].shape == (2, C) and out["feat"].shape == (2, 512)
    if m == "hcm":
        assert out["d"].shape == (2, C) and out["R"].shape == (2,)
        keys = set(net.state_dict())
        assert {"direction_head.weight", "magnitude_head.weight", "conv1.weight", "layer4.1.bn2.running_mean"} <= keys
print("[ok] models (HCM state_dict keys match the official ResNet18_32x32_HCM)")
print("all checks passed")

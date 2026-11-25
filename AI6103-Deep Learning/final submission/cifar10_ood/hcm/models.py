"""ResNet-18 for 32x32 inputs with a CE head or HCM heads.

The backbone and the HCM head names (`direction_head`, `magnitude_head`) are identical to
the official HCM repo (HCM_CIFAR10.py, class ResNet18_32x32_HCM), so official checkpoints
load with `model.load_state_dict(sd)` when `method="hcm"`.

Unified interface (both methods):
    out = model(x)                    -> dict with keys
        "logits": (B, C)  class scores used for accuracy (CE: fc output; HCM: R * d)
        "feat":   (B, 512) penultimate (avg-pooled) features
        HCM only:
        "d":      (B, C)  direction head output (NOT normalised)
        "R":      (B,)    magnitude head output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18_32x32(nn.Module):
    """ResNet-18 (CIFAR variant: 3x3 stem, no max-pool).

    method = "ce"  -> one linear head `fc` (num_classes)
    method = "hcm" -> `direction_head` (num_classes) + `magnitude_head` (1)
    """

    def __init__(self, method="ce", num_classes=10, num_blocks=(2, 2, 2, 2)):
        super().__init__()
        assert method in ("ce", "hcm"), method
        self.method = method
        self.num_classes = num_classes
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_size = 512
        if method == "ce":
            self.fc = nn.Linear(self.feature_size, num_classes)
        else:
            self.direction_head = nn.Linear(self.feature_size, num_classes)
            self.magnitude_head = nn.Linear(self.feature_size, 1)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)

    def forward_head(self, feat):
        if self.method == "ce":
            return {"logits": self.fc(feat), "feat": feat}
        d = self.direction_head(feat)
        R = self.magnitude_head(feat).squeeze(-1)
        return {"logits": d * R.unsqueeze(-1), "feat": feat, "d": d, "R": R}

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(method="ce", num_classes=10):
    return ResNet18_32x32(method=method, num_classes=num_classes)


def load_model(ckpt_path, map_location="cpu"):
    """Load a `final.pt` written by train.py. Returns (model, config_dict)."""
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["method"], cfg.get("num_classes", 10))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg

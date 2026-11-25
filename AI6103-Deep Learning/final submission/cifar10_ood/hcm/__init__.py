"""Shared library for the AI6103 HCM project (CIFAR-10 code).

Modules
-------
models   ResNet-18 (32x32) with either a CE head or the HCM direction/magnitude heads.
losses   Cross-entropy (hard or soft targets) and HCM losses (official script / paper Eq. 3).
mixup    Two-sample MixUp (Beta) and k-sample Dirichlet MixUp.
data     CIFAR-10 loaders and the six OOD test sets (same transforms as the official repo).
scores   HCM uncertainty formulas (paper Eq. 2 + the three variants in the official scripts).
metrics  AUROC / AUPR / FPR@95TPR with an explicit, documented convention.
utils    Seeding, checkpointing, JSON/CSV helpers.
"""

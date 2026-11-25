| Config | Seeds | Test acc (%) | HCM acc argmax d (%) | HCM OOD AUROC avg (%) | HCM FPR@95 avg (%) |
|---|---|---|---|---|---|
| ce | 5 | 94.07 ± 0.08 | – | – | – |
| hcm | 5 | 94.23 ± 0.18 | 94.23 ± 0.18 | 83.04 ± 0.29 | 59.94 ± 1.76 |
| ce_mix-k2-a1 | 5 | 94.85 ± 0.15 | – | – | – |
| hcm_mix-k2-a1 | 5 | 94.65 ± 0.14 | 94.65 ± 0.14 | 81.72 ± 2.63 | 56.71 ± 2.65 |
| ce_mix-k20-a0.5 | 3 | 84.97 ± 0.44 | – | – | – |
| hcm_mix-k20-a0.5 | 3 | 85.50 ± 0.40 | 85.50 ± 0.40 | 77.44 ± 1.16 | 80.14 ± 2.40 |
| hcm_mix-k2-a1-decomp | 5 | 94.81 ± 0.13 | 94.81 ± 0.13 | 84.90 ± 1.27 | 49.88 ± 2.17 |

mean ± sample std over seeds; last-epoch model; CIFAR-10 test set (10k).

# HCM vs. cross-entropy on CIFAR-10 (report Exp A; inputs of Exp B and Exp C)

Code for the CIFAR-10 part of our AI6103 project on *Uncertainty Estimation via Hyperspherical
Confidence Mapping* (HCM, ICLR 2026, OpenReview `G4JYxxI23T`). It trains ResNet-18 with
cross-entropy (CE) or HCM, with or without MixUp, and produces:

* **Exp A** (Table 1 of the report): test accuracy and HCM OOD detection for all variants.
* The exported logits / features / `d`, `R` of every model, which **Exp B** (MSP, MDS, KNN,
  Ensembles) and **Exp C** (score-formula comparison) read. Exp D retrains with `train.py`
  using the flags listed in §6.

**Where each experiment is in the final report**

| Experiment | Report | Code here |
|---|---|---|
| Exp A: reproduction and MixUp variants | Section 4.2, Table 1 | `train.py`, `export_outputs.py`, `eval_hcm_ood.py`, `aggregate_results.py` |
| Exp B: baselines (MSP, MDS, KNN, ensemble) | Section 4.3, Figure 2 | reads the exported `outputs/*.npz` (§5); the baseline scoring scripts are not in this folder |
| Exp C: score formulas | Section 4.4, Table 2 | `hcm/scores.py`, `eval_hcm_ood.py`; per-seed numbers in `results/results_hcm_ood.txt` |
| Exp D: training sensitivity | Section 4.5, Table 3 | `scripts/role3_*`, which call `train.py` with the flags of §6; see `SENSITIVITY.md` |

Model and loss code is a rewrite of the official implementation
(`github.com/Abandoned-Puppy/HCM`, commit `847486e`; the repository has no licence file).
The HCM heads use the official layer names, so official checkpoints load with `load_model`.

## 1. Setup

```bash
pip install -r requirements.txt              # torch>=2.1, torchvision, numpy, pillow, scikit-learn, matplotlib
python prepare_data.py --data_root ./data    # CIFAR-10/100, SVHN, MNIST, DTD (torchvision)
# Manual downloads (the OOD splits of the official repository):
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip -q tiny-imagenet-200.zip -d data/
wget http://data.csail.mit.edu/places/places365/val_256.tar && mkdir -p data/places365 && tar -xf val_256.tar -C data/places365
python prepare_data.py --data_root ./data    # TIN and Places365 should now be "found"
bash scripts/smoke_test.sh                   # ~5 min on random tensors; must end with SMOKE TEST PASSED
```

## 2. Reproduce Exp A

```bash
bash scripts/reproduce_expA.sh     # all 31 runs + export + evaluation + tables (~11.5 GPU-h on one A40)
```
or one run at a time (one run ≈ 22 min on an A40):

| Report row (Table 1) | Config (`run_queue.sh`) | `train.py` arguments | Seeds |
|---|---|---|---|
| CE | `ce` | `--method ce` | 0–4 |
| HCM | `hcm` | `--method hcm` | 0–4 |
| CE + MixUp | `ce_mix` | `--method ce --mix_k 2 --mix_alpha 1.0` | 0–4 |
| **HCM + MixUp (decomp, main)** | `hcm_mixdecomp` | `--method hcm --mix_k 2 --mix_alpha 1.0 --mix_target decomp` | 0–4 |
| HCM + MixUp (soft, comparison) | `hcm_mix` | `--method hcm --mix_k 2 --mix_alpha 1.0` | 0–4 |
| HCM + Dirichlet MixUp (k=20) | `hcm_mixdir` | `--method hcm --mix_k 20 --mix_alpha 0.5` | 0–2 |
| CE + Dirichlet MixUp (k=20) | `ce_mixdir` | `--method ce --mix_k 20 --mix_alpha 0.5` | 0–2 |

```bash
python train.py --method hcm --mix_k 2 --mix_alpha 1.0 --mix_target decomp --seed 0
CONFIGS="ce hcm" SEEDS="0 1 2 3 4" bash scripts/run_queue.sh
bash scripts/postprocess.sh        # export -> HCM OOD eval -> results/ tables and curves
```
On a SLURM cluster use `scripts/slurm_job.sbatch` (fill in partition, QoS and conda env first).
A run whose `final.pt` exists is skipped; an interrupted run resumes from `ckpt_last.pt`.

## 3. Reference results (`results/`)

Obtained with exactly this code (NTU CCDS TC2, one A40). Mean ± sample std over seeds,
last-epoch model, CIFAR-10 test set; OOD = average over CIFAR-100, TIN, MNIST, SVHN, DTD,
Places365 with the paper's score u(x) = R·| ‖d‖ − 1 |.

| Config | Seeds | Test acc. (%) | HCM OOD AUROC (%) | FPR@95 (%, OpenOOD) |
|---|---|---|---|---|
| CE | 5 | 94.07 ± 0.08 | – | – |
| HCM | 5 | 94.23 ± 0.18 | 83.04 ± 0.29 (paper 85.04) | 59.94 ± 1.76 |
| CE + MixUp | 5 | 94.85 ± 0.15 | – | – |
| HCM + MixUp (decomp, main) | 5 | 94.81 ± 0.13 | 84.90 ± 1.27 (paper 89.44) | 49.88 ± 2.17 |
| HCM + MixUp (soft) | 5 | 94.65 ± 0.14 | 81.72 ± 2.63 | 56.71 ± 2.65 |
| HCM + Dirichlet MixUp (k=20) | 3 | 85.50 ± 0.40 | 77.44 ± 1.16 | 80.14 ± 2.40 |
| CE + Dirichlet MixUp (k=20) | 3 | 84.97 ± 0.44 | – | – |

* `results/accuracy_summary.{md,csv}` – this table; `results/all_runs.csv` – one row per run.
* `results/results_hcm_ood.txt` – every HCM run × OOD set × score formula (AUROC / FPR@95).
* `results/test_acc_curves.png` – test accuracy and mean ‖d‖ per epoch.

Trained weights and exported outputs (≈8 GB, `runs/`) are on the team Google Drive:
<https://drive.google.com/drive/folders/1oDH3DbSpz56fgqzOhGHNa2Eus598iP5Z>

## 4. Implementation choices

Copied from the official script, or chosen and recorded where the paper is silent.

| Item | Choice |
|---|---|
| Backbone | ResNet-18, CIFAR variant (3×3 stem, no max-pool) |
| Optimisation | SGD, momentum 0.9, weight decay 1e-4, lr 0.1, ×0.1 at epochs 30/60/90, 100 epochs, batch 64 |
| Augmentation | RandomCrop(32, padding 4) + horizontal flip; official normalisation constants |
| Warm-up | loss ×0.1 in epochs 1–2 (official code, HCM only there); applied to CE too for a fair comparison |
| HCM loss | official CIFAR-10 script: MSE(d, d\*)^¾ + MSE(R·d\*, R\*·d\*)^¾, i.e. λ_norm = 0 (`--hcm_loss official`); paper Eq. 3 is `--hcm_loss paper` |
| MixUp | k = 2, λ ~ Beta(1, 1), one λ per batch, every batch mixed; CE + MixUp uses soft-target CE |
| Mixed label → (R\*, d\*) | `soft`: d\* = y, R\* = 1. `decomp`: R\* = ‖y‖, d\* = y/‖y‖. `decomp` was chosen as the main setting **after** seeing the OOD results, so both are reported |
| Dirichlet MixUp | k = 20, w ~ Dir(0.5), every batch; drops accuracy to ≈85 % (negative result) |
| Prediction | HCM: argmax(R·d) (`test_acc`); argmax(d) also logged (`test_acc_dir`) |
| Model selection | last epoch, no validation split |
| OOD sets | official-repository splits (all TIN test images, Places365 validation images), not OpenOOD's curated lists |
| FPR@95 | OpenOOD convention (ID positive). The official code's opposite convention is stored as `fpr95_hcm_repo` |

## 5. Output format (what Exp B / Exp C read)

```
runs/<config>_s<seed>/
  config.json  log.csv  metrics.json  final.pt  ood_hcm.json (HCM only)
  outputs/{train,test,CIFAR100,TIN,MNIST,SVHN,DTD,PLACES365}.npz
     labels (N,) [-1 for OOD] · logits (N,10) · feat (N,512) · HCM only: d (N,10), R (N,)
```
`train.npz` is the 50k training set without augmentation (for MDS class statistics and the
KNN bank). Load a model with `from hcm.models import load_model; model, cfg = load_model("runs/hcm_s0/final.pt")`.
Use `hcm.metrics.ood_metrics(id_uncertainty, ood_uncertainty)` for all methods (higher = more
OOD-like; for MSP pass `-max_softmax`), so AUROC and FPR@95 are defined identically everywhere.

## 6. Other flags (used in Exp D)

`--lr`, `--warmup_mode {loss_scale,linear_lr,none}`, `--warmup_epochs`, `--warmup_factor`,
`--lambda_norm`, `--hcm_loss {official,paper}`, `--loss_power`, `--mix_target {soft,decomp}`,
`--amp` (bf16), `--deterministic`. Run folders encode non-default values automatically; a run
whose loss becomes non-finite is stopped and recorded as `"diverged": true` in `metrics.json`.
Score formulas live in `hcm/scores.py` (`paper_eq2`, `official_cifar`, `official_toy`,
`official_moons`, `norm_dev`); `python eval_hcm_ood.py --runs "runs/hcm*"` evaluates all of them.

## 7. Files

```
train.py              training (CE / HCM, ± MixUp, resume, divergence detection)
export_outputs.py     logits / features / d / R for train, test and the six OOD sets
eval_hcm_ood.py       HCM OOD metrics for every score formula
aggregate_results.py  summary tables over seeds
plot_curves.py        accuracy and ‖d‖ curves
prepare_data.py       dataset download
hcm/                  models, losses, mixup, data, scores, metrics, utils
scripts/              reproduce_expA.sh, run_queue.sh, postprocess.sh, smoke_test.sh, slurm_job.sbatch
                      role3_* and tc2_job.sbatch: the sensitivity runs (SENSITIVITY.md)
tests/check_core.py   loss equals the official script, MixUp sanity, model keys
results/              reference results of our runs
```

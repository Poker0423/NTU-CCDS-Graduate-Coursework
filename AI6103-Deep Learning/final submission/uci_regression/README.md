# HCM on UCI scalar regression (Section 5 of the report)

Everything runs on a laptop CPU. One training run takes 1 to 20 seconds; the largest
group of runs (training sensitivity, 2100 runs) takes about 40 minutes.

## Files

| File | Purpose |
|---|---|
| `data.py` | Loads the six UCI datasets (wine, concrete, energy, kin8nm, power, yacht) and the two-target `energy2`; 80/10/10 split |
| `models.py` | MLP, HCM network, HCM network with a prediction head (`HCMPlus`), Deep Ensemble member, HCM loss |
| `train.py` | Trains one model and saves per-sample outputs (`.npz`) and metrics (`.json`) |
| `run_grid.py` | Runs a grid of configurations in parallel |
| `analyze_c.py`, `analyze_d.py`, `analyze_e.py` | Offline analysis of the score (C), the intervals (D) and the training sensitivity (E); they read `results/` and do not retrain |
| `analyze_f.py`, `analyze_f_modes.py` | Offline analysis of the two improvements (F) |
| `make_figures.py` | Writes all figures to `figures/` and all tables to `tables/` |
| `summarize.py` | Prints an RMSE summary of one result folder |
| `data_cache/` | The datasets, already downloaded |
| `figures/`, `tables/` | The outputs of `make_figures.py` from our runs (tables kept as `.csv`; `make_figures.py` also writes a LaTeX `.tex` version of each) |

## Environment

```bash
pip install -r requirements.txt
```

Versions we used: Python 3.8, torch 2.3, numpy 1.24, scipy 1.10, scikit-learn 1.3,
pandas 2.0, matplotlib 3.5. Newer versions were not tested.

## Setup

From Appendix A.1 of the HCM paper: MLP with 3 hidden layers of 20 units,
LeakyReLU(0.01); Adam with betas (0.9, 0.999) and weight decay 1e-4; 200 epochs;
80/10/10 split; Gaussian noise with std 0 to 5 added to the test inputs. A scalar target
`y` is embedded as `(y, y)`, as in Section 2.2 of the paper.

Details that the paper does not state and that we chose:

| Item | Our choice |
|---|---|
| Batch size | 64 |
| Inputs | standardized with the training mean and std; test noise is added after standardization |
| Evaluated weights | last epoch (no early stopping) |
| Target preprocessing | `--target_mode`: `raw`, `standardize`, `center`, `scale:<c>` |
| Calibration set | the 10% validation split |
| Deep Ensemble | 5 members, Gaussian NLL |
| MC Dropout | dropout 0.1, 50 forward passes at test time |

**Main setting:** standardized targets, learning rate 1e-3, lambda 0, no warm-up.
**Paper setting:** raw targets, learning rate 1e-4. It does not finish training on
concrete, energy and yacht and is used only as a reference.

Loss (`models.hcm_loss`): `MSE(R_hat * d, y) + MSE(R * d_hat, y) + lambda * (||d_hat|| - 1)^2`.
The first two terms are those of the official `Toy_example.py`.
Score: `u = |R_hat| * | ||d_hat|| - 1 |` (Eq. 2 of the paper).

### The two improvements (Section 5.5)

| `--method` | What it does |
|---|---|
| `hcmp` | HCM heads plus a prediction head trained with MSE. The prediction comes from the prediction head; the score is still `u` |
| `rhcm` | Residual-origin HCM: the HCM heads decompose the standardized residual `(t - f(x)) / s`, where `f` is the prediction head and `s` is a running root-mean-square of the training residuals. The score is `|R_hat|` |

`--branch` sets how the HCM heads use the trunk: `shared` (trained jointly), `detach`
(the heads read the trunk features but send no gradient back), `own` (a separate trunk).

## Reproduction

Each command fills one folder under `results/`. `--resume` skips runs that already have
a `.json` file.

```bash
# Basic comparison of MLP and HCM (Section 5.2)
python run_grid.py --exp step1 --methods mlp --target_modes raw standardize --lrs 1e-4 1e-3
python run_grid.py --exp step2 --methods hcm --target_modes raw standardize --lrs 1e-4 1e-3

# C: what the score measures; five target preprocessings, 5 seeds (Section 5.3, Figure 3)
python run_grid.py --exp stepC --methods hcm --datasets wine concrete energy kin8nm power yacht energy2 \
  --target_modes raw standardize center scale:0.1 scale:10 --lrs 1e-4 1e-3 --seeds 0 1 2 3 4

# D: prediction intervals; 20 seeds (Section 5.4, Table 4)
python run_grid.py --exp stepD --methods hcm --target_modes standardize raw --lrs 1e-3 --seeds $(seq 0 19)
python run_grid.py --exp stepD --methods mlp mcd ens --target_modes standardize --lrs 1e-3 --seeds $(seq 0 19)

# E: learning rate, norm penalty, warm-up, batch size (Sections 5.2 and 5.5)
python run_grid.py --exp stepE --methods hcm --target_modes standardize raw \
  --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 --lams 0 0.1 1 3 5 --seeds 0 1 2 3 4
python run_grid.py --exp stepE --methods hcm --target_modes standardize raw \
  --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 --warmups 5 20 --seeds 0 1 2 3 4
python run_grid.py --exp stepE_bs8 --methods hcm --datasets concrete energy yacht --batch 8
python run_grid.py --exp stepE_bs1 --methods hcm --datasets concrete energy yacht --batch 1

# F: the two improvements; 20 seeds, then four other target preprocessings (Section 5.5, Table 5, Figure 4)
python run_grid.py --exp stepF --methods hcmp rhcm --branches shared detach own \
  --target_modes standardize --lrs 1e-3 --seeds $(seq 0 19)
python run_grid.py --exp stepF_modes --methods rhcm --branches detach own \
  --target_modes raw center scale:0.1 scale:10 --lrs 1e-3 --seeds 0 1 2 3 4
```

Then run the analysis and draw the figures and tables:

```bash
python analyze_c.py
python analyze_d.py
python analyze_e.py
python analyze_f.py
python analyze_f_modes.py
python make_figures.py            # everything; or a subset: python make_figures.py c1 d2 tables
```

## Result files

Each run saves two files named
`<dataset>_<method>_<target mode>_lr<lr>_lam<lambda>_wu<warm-up epochs>_s<seed>`
(for `hcmp` and `rhcm` the method name carries the branch, for example `rhcm-detach`).

- `.json`: configuration and summary metrics (RMSE per split, AUROC, Spearman, `diverged`).
- `.npz`: per-sample outputs. Splits are `train`, `val`, `test_s0` … `test_s5` (the number
  is the noise std). Each split has `<split>_y` (truth, original units) and
  `<split>_yhat` (prediction, original units). HCM also saves `<split>_R` and
  `<split>_dnorm` in the transformed target space; Deep Ensemble and MC Dropout save
  `<split>_sigma` (original units). `y_shift` and `y_scale` define the target transform
  `t = (y - shift) / scale`.

The HCM score in original units is `u = |R| * |dnorm - 1| * y_scale`. For `rhcm`, the
saved `_R` already includes the residual scale `s`, so the residual-origin score in
original units is `R_r = |R| * y_scale`.

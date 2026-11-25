# Training sensitivity (Exp D; Section 4.5 and Table 3 of the report)

The sensitivity runs use the same training and evaluation code as Exp A (`train.py`,
`export_outputs.py`, `eval_hcm_ood.py`, `hcm/`). The scripts below define the ten settings
and summarize them. They are the versions that were run on NTU CCDS TC2 on 2026-11-19.

## Scripts

| File | Purpose |
|---|---|
| `scripts/role3_sensitivity.sh` | Entry point: `train`, `eval` or `all` |
| `scripts/role3_sensitivity_train.sh` | Defines the ten settings and trains them one after another |
| `scripts/role3_sensitivity_eval.sh` | Exports the outputs of each final model, computes the OOD metrics for the five score formulas, then summarizes and archives |
| `scripts/role3_summarize_task_e.py` | Checks the state of every run, builds the summary tables, and can recompute the metrics from the raw outputs (`--verify-metrics`) |
| `scripts/role3_collect.py` | Collects logs, configurations, metrics and a file manifest into one archive (no checkpoints) |
| `scripts/role3_prepare_data.py`, `.sh` | Downloads and validates the datasets; the `.sh` wrapper has the TC2 environment of the original run |
| `scripts/tc2_job.sbatch` | The SLURM job as it was submitted, including the account check, modules and environment of the original run; edit these before reuse |

## Settings

Baseline: CIFAR-10, ResNet-18, HCM with pairwise MixUp (`k=2`, `alpha=1.0`) and decomposed
targets, seed 0, 100 epochs, SGD with learning rate 0.1. The loss is the `official` form
with exponent 0.75 and `lambda_norm=0`. In the first two epochs the loss is multiplied by
0.1; this is a scaling of the loss, not a learning-rate warm-up. The last-epoch model is
evaluated. Each setting changes one item:

| Label | Change from the baseline | Row in Table 3 |
|---|---|---|
| `baseline` | none (the baseline is retrained) | Baseline |
| `lr_0p05`, `lr_0p2` | learning rate 0.05, 0.2 | Learning rate 0.05, 0.2 |
| `warmup_5`, `warmup_none` | loss scaling for 5 epochs, no loss scaling | Warm-up 5 epochs, No warm-up |
| `lambda_0p1`, `lambda_1` | norm penalty weight 0.1, 1.0 | lambda_norm = 0.1, 1.0 |
| `loss_power_1` | exponent of the official loss set to 1 | Loss power 1 |
| `mix_target_soft` | soft instead of decomposed MixUp targets | Soft MixUp targets |
| `hcm_loss_paper` | the `paper` branch of `hcm/losses.py` (exponent 2) | Paper-form loss |

Nine runs finished and one diverged. Every setting uses one seed. The summary script uses
the `official_cifar` score for its main table and averages over the six OOD sets with
equal weights; the other four formulas are kept in the full tables.

## Note on the `paper` loss branch

`--hcm_loss paper` weights the direction term with the **predicted** magnitude,
`phi(R_hat * ||d_hat - d*||)`, see `hcm/losses.py`. Eq. 3 of the HCM paper weights it with
the **true** magnitude. This run recorded a non-finite loss in the third iteration of the
first epoch and completed no epoch. It therefore does not show that the loss of Eq. 3
diverges; a strict test needs a corrected implementation and a new run.

## How to run

Prepare the data first (see `README.md`, §1) and use a GPU environment.

```bash
python tests/check_core.py                    # optional CPU check of the core code
bash scripts/role3_sensitivity.sh train       # trains the ten settings (several hours)
bash scripts/role3_sensitivity.sh eval        # export, evaluation, summary, archive
```

Outputs go to `runs_role3/`, `results_role3/` and `role3_task_e_results.tar.gz`.
`OUT_DIR` and `DATA_ROOT` change the run and data folders, but `role3_collect.py` only
collects the default `runs_role3/`. Evaluation skips a run without `final.pt` and records
its state in the summary.

On a SLURM cluster, edit the partition, QoS, account check, modules and environment in
`scripts/tc2_job.sbatch`, then:

```bash
mkdir -p logs
sbatch --export=ALL,ROLE3_STAGE=train scripts/tc2_job.sbatch
sbatch --export=ALL,ROLE3_STAGE=eval scripts/tc2_job.sbatch    # after training has finished
```

## Not in this package

The training data, the nine `final.pt` checkpoints and the exported outputs of these runs.
Without them the reported metrics cannot be recomputed from the raw model outputs.

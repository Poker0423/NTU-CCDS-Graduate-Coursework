#!/usr/bin/env bash
# After training: export logits/features/d/R (inputs of Exp B baselines and Exp C score
# formulas), HCM OOD evaluation, accuracy tables and curves.
#   TIN_DIR=./data/tiny-imagenet-200/test/images PLACES_DIR=./data/places365 bash scripts/postprocess.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_DIR=${OUT_DIR:-./runs}
DATA_ROOT=${DATA_ROOT:-./data}
TIN_DIR=${TIN_DIR:-$DATA_ROOT/tiny-imagenet-200/test/images}
PLACES_DIR=${PLACES_DIR:-$DATA_ROOT/places365}

python export_outputs.py --runs "$OUT_DIR/*" --data_root "$DATA_ROOT" --tin_dir "$TIN_DIR" --places_dir "$PLACES_DIR"
python eval_hcm_ood.py   --runs "$OUT_DIR/*" | tee results_hcm_ood.txt
python aggregate_results.py --runs_dir "$OUT_DIR" --out_dir results
python plot_curves.py       --runs_dir "$OUT_DIR" --out results/test_acc_curves.png

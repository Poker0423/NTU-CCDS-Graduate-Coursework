#!/usr/bin/env bash
# End-to-end smoke test on random tensors (no download, CPU or GPU, ~2-5 min).
# Run this first on any new machine:   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=./runs_smoke
rm -rf "$OUT" results_smoke
python tests/check_core.py
COMMON="--fake_data --epochs 3 --max_steps 4 --workers 0 --out_dir $OUT --print_every 0"
python train.py --method ce  $COMMON
python train.py --method hcm $COMMON
python train.py --method ce  --mix_k 2 --mix_alpha 1.0 $COMMON
python train.py --method hcm --mix_k 2 --mix_alpha 1.0 $COMMON
python train.py --method hcm --mix_k 20 --mix_alpha 0.5 $COMMON
python train.py --method hcm --hcm_loss paper --lambda_norm 1 --warmup_mode linear_lr $COMMON
python train.py --method hcm $COMMON   # second call must print [skip]
python export_outputs.py --runs "$OUT/*" --fake_data --workers 0 --oods CIFAR100 SVHN
python eval_hcm_ood.py --runs "$OUT/*" > /dev/null
python aggregate_results.py --runs_dir "$OUT" --out_dir results_smoke
python plot_curves.py --runs_dir "$OUT" --out results_smoke/curves.png
echo "SMOKE TEST PASSED (accuracies on random data are meaningless)"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")"
OUT_DIR=${OUT_DIR:-./runs_role3}
DATA_ROOT=${DATA_ROOT:-./data}
mkdir -p "$OUT_DIR/_logs"

evaluate_one() {
  local label=$1
  local run_name="hcm_mix-k2-a1-decomp_s0_role3_${label}"
  local run_dir="$OUT_DIR/$run_name"
  local log="$OUT_DIR/_logs/${label}.log"
  echo "=== $(date '+%F %T') evaluate=$label ===" | tee -a "$log"
  if [[ ! -f "$run_dir/final.pt" ]]; then
    echo "[skip eval] missing final.pt; metrics follow when available:" | tee -a "$log"
    if [[ -f "$run_dir/metrics.json" ]]; then cat "$run_dir/metrics.json" | tee -a "$log"; fi
    return 0
  fi
  python -u export_outputs.py --runs "$run_dir" --data_root "$DATA_ROOT" \
    --batch 256 --no_train --oods CIFAR100 TIN MNIST SVHN DTD PLACES365 \
    2>&1 | tee -a "$log"
  python -u eval_hcm_ood.py --runs "$run_dir" 2>&1 | tee -a "$log"
}

evaluate_one baseline
evaluate_one lr_0p05
evaluate_one lr_0p2
evaluate_one warmup_5
evaluate_one warmup_none
evaluate_one lambda_0p1
evaluate_one lambda_1
evaluate_one hcm_loss_paper
evaluate_one loss_power_1
evaluate_one mix_target_soft

echo "=== sensitivity evaluation finished $(date '+%F %T') ==="
set +e
python scripts/role3_summarize_task_e.py --runs-root "$OUT_DIR" --out ./results_role3 --project . --verify-metrics
summary_status=$?
set -e
python scripts/role3_collect.py
exit "$summary_status"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")"
OUT_DIR=${OUT_DIR:-./runs_role3}
DATA_ROOT=${DATA_ROOT:-./data}
mkdir -p "$OUT_DIR/_logs"

run_one() {
  local label=$1
  shift
  local run_name="hcm_mix-k2-a1-decomp_s0_role3_${label}"
  local run_dir="$OUT_DIR/$run_name"
  local log="$OUT_DIR/_logs/${label}.log"
  echo "=== $(date '+%F %T') sensitivity=$label ===" | tee -a "$log"
  python -u train.py \
    --method hcm --mix_k 2 --mix_alpha 1.0 --mix_target decomp --seed 0 \
    --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" --run_name "$run_name" "$@" \
    2>&1 | tee -a "$log"
  if [[ ! -f "$run_dir/final.pt" && -f "$run_dir/metrics.json" ]]; then
    echo "Training ended without final.pt; metrics follow:" | tee -a "$log"
    cat "$run_dir/metrics.json" | tee -a "$log"
  fi
}

run_one baseline
run_one lr_0p05 --lr 0.05
run_one lr_0p2 --lr 0.2
run_one warmup_5 --warmup_mode loss_scale --warmup_epochs 5 --warmup_factor 0.1
run_one warmup_none --warmup_mode none
run_one lambda_0p1 --lambda_norm 0.1
run_one lambda_1 --lambda_norm 1.0
run_one hcm_loss_paper --hcm_loss paper
run_one loss_power_1 --loss_power 1.0
run_one mix_target_soft --mix_target soft

echo "=== sensitivity training finished $(date '+%F %T') ==="

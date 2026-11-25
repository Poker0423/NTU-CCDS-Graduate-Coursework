#!/usr/bin/env bash
# Run a list of configs x seeds one after another.
# Finished runs are skipped and interrupted runs resume, so a job killed by a time
# limit can simply be started again.
#
#   bash scripts/run_queue.sh                                   # main Exp A configs, seeds 0-4
#   CONFIGS="ce hcm" SEEDS="0 1 2" bash scripts/run_queue.sh
#   CONFIGS="hcm_mixdecomp" SEEDS="0" EXTRA="--amp" bash scripts/run_queue.sh
#
# Config name -> train.py arguments -> run-folder prefix (= row in the report, Exp A)
#   ce            -> ce_*                    CE
#   hcm           -> hcm_*                   HCM
#   ce_mix        -> ce_mix-k2-a1_*          CE + MixUp
#   hcm_mixdecomp -> hcm_mix-k2-a1-decomp_*  HCM + MixUp (decomp, main)
#   hcm_mix       -> hcm_mix-k2-a1_*         HCM + MixUp (soft, comparison)
#   hcm_mixdir    -> hcm_mix-k20-a0.5_*      HCM + Dirichlet MixUp (k=20)
#   ce_mixdir     -> ce_mix-k20-a0.5_*       CE + Dirichlet MixUp (k=20)
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A ARGS=(
  [ce]="--method ce"
  [hcm]="--method hcm"
  [ce_mix]="--method ce  --mix_k 2 --mix_alpha 1.0"
  [hcm_mix]="--method hcm --mix_k 2 --mix_alpha 1.0"
  [ce_mixdir]="--method ce  --mix_k 20 --mix_alpha 0.5"
  [hcm_mixdir]="--method hcm --mix_k 20 --mix_alpha 0.5"
  [hcm_mixdecomp]="--method hcm --mix_k 2 --mix_alpha 1.0 --mix_target decomp"
)

CONFIGS=${CONFIGS:-"ce hcm ce_mix hcm_mixdecomp hcm_mix"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATA_ROOT=${DATA_ROOT:-./data}
OUT_DIR=${OUT_DIR:-./runs}
EXTRA=${EXTRA:-}

mkdir -p "$OUT_DIR/_logs"
for seed in $SEEDS; do
  for c in $CONFIGS; do
    [[ -n "${ARGS[$c]:-}" ]] || { echo "unknown config $c"; exit 1; }
    echo "=== $(date '+%F %T') config=$c seed=$seed ==="
    python train.py ${ARGS[$c]} --seed "$seed" --data_root "$DATA_ROOT" --out_dir "$OUT_DIR" $EXTRA \
      2>&1 | tee -a "$OUT_DIR/_logs/${c}_s${seed}.log"
  done
done
echo "=== queue finished $(date '+%F %T') ==="

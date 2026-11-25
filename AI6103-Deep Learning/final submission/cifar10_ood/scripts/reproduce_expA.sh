#!/usr/bin/env bash
# Reproduce every row of the report's Exp A tables, then build the summary tables.
# 31 runs x ~22 min on one A40 (~11.5 GPU-hours). Re-run after an interruption: finished
# runs are skipped and the interrupted one resumes.
#   bash scripts/reproduce_expA.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIGS="ce hcm ce_mix hcm_mixdecomp hcm_mix" SEEDS="0 1 2 3 4" bash scripts/run_queue.sh
CONFIGS="hcm_mixdir ce_mixdir"                SEEDS="0 1 2"     bash scripts/run_queue.sh
bash scripts/postprocess.sh

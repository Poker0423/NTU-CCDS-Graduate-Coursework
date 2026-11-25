#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "$HOME/envs/ai6103"
set -u
export PYTHONUNBUFFERED=1
exec python -u scripts/role3_prepare_data.py --project "$PWD"

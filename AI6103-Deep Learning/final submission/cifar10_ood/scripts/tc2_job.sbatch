#!/bin/bash
#SBATCH --job-name=role3_sensitivity
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

if [[ "${USER:-}" != "zhongda002" ]]; then
  echo "Expected TC2 account zhongda002; got ${USER:-unknown}" >&2
  exit 1
fi

module load anaconda
module load cuda/12.8.0
eval "$(conda shell.bash hook)"
conda activate "$HOME/envs/ai6103"
set -u
# sbatch --chdir selects this project; SLURM_SUBMIT_DIR may be the SSH home.
[[ -f train.py ]] || { echo "Expected project working directory: $PWD" >&2; exit 1; }
mkdir -p logs

echo "account=$USER host=$(hostname) python=$(command -v python) start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0))'
python -m pip freeze > "logs/environment_${SLURM_JOB_ID}.txt"
ROLE3_STAGE=${ROLE3_STAGE:-train}
bash scripts/role3_sensitivity.sh "$ROLE3_STAGE"
echo "finished=$(date)"

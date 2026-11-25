#!/bin/bash
# Generic SLURM job (this is the script all Exp A runs were trained with on NTU CCDS TC2).
# Fill in the <...> lines for your own account, then from the package root e.g.:
#   sbatch --export=ALL,SEEDS="0 1 2" scripts/slurm_job.sbatch                     # train, then postprocess
#   sbatch --export=ALL,CONFIGS="hcm_mixdecomp",SEEDS="3 4" scripts/slurm_job.sbatch
#   sbatch scripts/slurm_job.sbatch                                                # postprocess only
# A job stopped by the time limit can be submitted again: it resumes.
#SBATCH --job-name=hcm_cifar10
#SBATCH --partition=<PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# module load anaconda cuda          # <- your cluster's modules
eval "$(conda shell.bash hook)"
conda activate <YOUR_ENV>
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname) | start $(date) | CONFIGS=[${CONFIGS:-default}] SEEDS=[${SEEDS:-}]"
if [ -n "${SEEDS:-}" ]; then bash scripts/run_queue.sh; fi
bash scripts/postprocess.sh
echo "end $(date)"

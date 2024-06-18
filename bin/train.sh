#!/bin/sh

#SBATCH --job-name=test-turpan-shared
#SBATCH --output=ML-%j-gnn.out
#SBATCH --error=ML-%j-gnn.err
#SBATCH -p shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

export MASTER_PORT=$(echo "${SLURM_JOB_ID} % 100000 % 50000 + 30000" | bc)
export MASTER_ADDR=$(hostname --ip-address)
echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.04-calmip-si.sif python train_script.py

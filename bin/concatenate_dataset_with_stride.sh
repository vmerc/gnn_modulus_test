#!/bin/sh
#SBATCH --job-name=test-turpan-shared
#SBATCH --output=ML-%j-gnn.out
#SBATCH --error=ML-%j-gnn.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1

export MASTER_PORT=$(echo "${SLURM_JOB_ID} % 100000 % 50000 + 30000" | bc)
export MASTER_ADDR=$(hostname --ip-address)

# Ensure RANK is set, defaulting to 0 if not provided
export RANK=${SLURM_PROCID:-0}
# Ensure WORLD_SIZE is set, defaulting to 1 if not provided
export WORLD_SIZE=${SLURM_NTASKS:-1}
# Print environment variables for debugging
echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"
echo "RANK=${RANK}"

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif concatenate_dataset_with_stride.py
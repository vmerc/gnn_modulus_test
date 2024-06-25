#!/bin/sh
#SBATCH --job-name=test-turpan-shared
#SBATCH --output=ML-%j-gnn.out
#SBATCH --error=ML-%j-gnn.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
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

# Check if a config name was provided
if [ -z "$1" ]; then
  echo "No config name provided. Usage: ./your_script.sh <config_name>"
  exit 1
fi

CONFIG_NAME=$1

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python train_script_double_encoder.py $CONFIG_NAME

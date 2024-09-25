#!/bin/bash
#SBATCH --job-name=test-turpan-small
#SBATCH --output=ML-%j-gnn.out
#SBATCH --error=ML-%j-gnn.err

#SBATCH -N 5
#SBATCH -n 10
#SBATCH --gpus-per-node=2
#SBATCH -p small
#SBATCH --ntasks-per-node=2
# Check if a config name was provided
if [ -z "$1" ]; then
  echo "No config name provided. Usage: ./your_script.sh <config_name>"
  exit 1
fi

CONFIG_NAME=$1
module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
# Execute the training script within the container
srun apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python train_script.py $CONFIG_NAME


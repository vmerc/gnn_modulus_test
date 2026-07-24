#!/bin/bash
#SBATCH --output=%x-%j-gnn.out
#SBATCH --error=%x-%j-gnn.err
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -p small
#SBATCH --ntasks-per-node=1

if [ -z "$1" ]; then
  echo "No config name provided. Usage: ./train_push_source_nodes.sh <config_name>"
  exit 1
fi

CONFIG_NAME=$1
module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
srun apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python train_push_source_nodes.py $CONFIG_NAME

#!/bin/bash
#SBATCH --output=%x-%j-gnn.out
#SBATCH --error=%x-%j-gnn.err
#SBATCH -N 4
#SBATCH --gpus-per-node=2
#SBATCH -p small
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --time=04:00:00

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: train_push_source_nodes.sh CONFIG DATASET_ROOT [INIT_CHECKPOINT_DIR]" >&2
  exit 1
fi

export TET_GHOST_DATA_ROOT="$2"
if [ "$#" -eq 3 ]; then
  export TET_GHOST_INIT_CKPT="$3"
fi

module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
srun apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python -u train_push_source_nodes.py "$1"

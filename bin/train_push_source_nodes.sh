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

submit_dir=${SLURM_SUBMIT_DIR:?Répertoire de soumission Slurm absent}
if [ -d "$submit_dir/conf/papier_ghost" ]; then
  bin_dir=$submit_dir
else
  bin_dir=$submit_dir/bin
fi
cd -- "$bin_dir"

if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch train_push_source_nodes.sh papier_ghost/E1" >&2
  exit 1
fi

module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
srun --kill-on-bad-exit=1 apptainer exec --bind /tmpdir,/work --pwd "$bin_dir" --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python -u train_push_source_nodes.py "$1"

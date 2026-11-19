#!/bin/bash
#SBATCH --job-name=prepare-ghost-graphs
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

submit_dir=${SLURM_SUBMIT_DIR:?Répertoire de soumission Slurm absent}
if [ -d "$submit_dir/conf/papier_ghost" ]; then
  bin_dir=$submit_dir
else
  bin_dir=$submit_dir/bin
fi
cd -- "$bin_dir"

srun apptainer exec --bind /tmpdir,/work,/users --pwd "$bin_dir" --nv \
  /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python -u prepare_tet_ghost_graphs.py

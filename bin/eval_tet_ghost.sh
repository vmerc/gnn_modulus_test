#!/usr/bin/env bash
#SBATCH --job-name=eval-tet-ghost
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

submit_dir=${SLURM_SUBMIT_DIR:?Répertoire de soumission Slurm absent}
if [ -d "$submit_dir/conf/papier_ghost" ]; then
  bin_dir=$submit_dir
else
  bin_dir=$submit_dir/bin
fi
cd -- "$bin_dir"
image=/work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif

srun apptainer exec --bind /tmpdir,/work --pwd "$bin_dir" --nv "$image" \
    python -u eval_tet_ghost.py "$@"

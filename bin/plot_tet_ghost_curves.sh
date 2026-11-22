#!/usr/bin/env bash
#SBATCH --job-name=plot-tet-curves
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -euo pipefail

submit_dir=${SLURM_SUBMIT_DIR:?Répertoire de soumission Slurm absent}
if [ -d "$submit_dir/conf/papier_ghost" ]; then
  bin_dir=$submit_dir
else
  bin_dir=$submit_dir/bin
fi
cd -- "$bin_dir"

git_commit=${TURPAN_COMMIT:-$(git -C "$bin_dir/.." rev-parse --short HEAD)}
echo "Commit: $git_commit"
echo "Entrypoint: bin/plot_tet_ghost_curves.py"

srun apptainer exec --bind /tmpdir,/work,/users --pwd "$bin_dir" \
  /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python -u plot_tet_ghost_curves.py "$@"

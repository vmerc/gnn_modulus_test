#!/usr/bin/env bash
#SBATCH --job-name=eval-tet-ghost
#SBATCH --output=%x-%j-eval.out
#SBATCH --error=%x-%j-eval.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

submit_dir=${SLURM_SUBMIT_DIR:?Répertoire de soumission Slurm absent}
if [ -d "$submit_dir/conf/papier_ghost" ]; then
  bin_dir=$submit_dir
else
  bin_dir=$submit_dir/bin
fi
cd -- "$bin_dir"
image=/work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: sbatch eval_tet_ghost.sh papier_ghost/E1 [EPOCH]" >&2
  exit 1
fi

git_commit=${TURPAN_COMMIT:-$(git -C "$bin_dir/.." rev-parse --short HEAD)}
echo "Commit: $git_commit"
echo "Entrypoint: bin/eval_tet_ghost.py"
echo "Configuration: $1"
echo "Epoch: ${2:-900}"

srun apptainer exec --bind /tmpdir,/work --pwd "$bin_dir" --nv "$image" \
    python -u eval_tet_ghost.py "$@"

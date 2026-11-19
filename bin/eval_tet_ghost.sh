#!/usr/bin/env bash
#SBATCH --job-name=eval-tet-ghost
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

cd -- "$(dirname -- "$0")"
image=/work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif

srun apptainer exec --bind /tmpdir,/work --nv "$image" \
    python -u eval_tet_ghost.py "$@"

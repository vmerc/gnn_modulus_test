#!/bin/sh
#SBATCH --job-name=prepare-x8-ghost
#SBATCH --output=prepare_x8_ghost-%j.out
#SBATCH --error=prepare_x8_ghost-%j.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:0

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python "$SCRIPT_DIR/prepare_x8_short_ghost_dataset.py"

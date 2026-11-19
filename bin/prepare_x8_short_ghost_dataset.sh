#!/bin/sh
#SBATCH --job-name=prepare-x8-ghost
#SBATCH --output=prepare_x8_ghost-%j.out
#SBATCH --error=prepare_x8_ghost-%j.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4

set -eu

cd -- "${SLURM_SUBMIT_DIR:?Soumettre ce job depuis le dossier bin}"
srun apptainer exec --bind /tmpdir,/work \
  --pwd "$SLURM_SUBMIT_DIR" \
  /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python -u prepare_x8_short_ghost_dataset.py

#!/bin/sh
#SBATCH --job-name=prepare-aube-t1-ghost
#SBATCH --output=prepare_aube_t1_ghost-%j.out
#SBATCH --error=prepare_aube_t1_ghost-%j.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:0


apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python "prepare_aube_T1_ghost_dataset.py"

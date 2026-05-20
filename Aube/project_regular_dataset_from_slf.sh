#!/bin/sh
#SBATCH --job-name=aube-regular-proj
#SBATCH --output=aube-regular-proj-%j.out
#SBATCH --error=aube-regular-proj-%j.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:0

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python project_regular_dataset_from_slf.py

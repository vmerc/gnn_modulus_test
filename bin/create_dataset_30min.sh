#!/bin/sh
#SBATCH --job-name=test-turpan-shared
#SBATCH --output=create_dataset_30.out
#SBATCH --error=create_dataset_30.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:0

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python create_dataset_30min.py

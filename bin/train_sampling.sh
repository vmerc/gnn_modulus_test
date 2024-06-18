#!/bin/sh

#SBATCH --job-name=test-turpan-shared
#SBATCH --output=ML-%j-gnn.out
#SBATCH --error=ML-%j-gnn.err
#SBATCH -p shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.04-calmip-si.sif python train_script_sampling.py

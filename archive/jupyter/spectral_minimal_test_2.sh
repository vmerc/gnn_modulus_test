#!/bin/bash
#SBATCH --output=%x-%j-gnn.out
#SBATCH --error=%x-%j-gnn.err
#SBATCH --dependency=singleton
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gpus-per-node=2
#SBATCH -p small
#SBATCH --ntasks-per-node=2
#SBATCH --dependency=singleton   # job dependency
# Check if a config name was provided
module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
# Execute the training script within the container

CONFIG_NAME="Config1Group3Shortx4MultiTest"
SAVING_NAME="x8"
MESH_LIST="/work/m24046/m24046mrcr/results_data_30min_35_70_maillagex8/Mesh8_corrige.slf"

srun apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python spectral_minimal_test.py --config_name "$CONFIG_NAME" --saving_name "$SAVING_NAME" --mesh_list "$MESH_LIST" 


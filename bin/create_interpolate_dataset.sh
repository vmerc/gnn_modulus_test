#!/bin/sh
#SBATCH --job-name=interpolate-shared
#SBATCH --output=interpolate-shared-%j.out
#SBATCH --error=interpolate-shared-%j.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:1
# Shell script to launch the interpolate_mesh.py script with specified .res files

# Set the paths to the fine and coarse mesh directories
NUMBER=$1
FINE_MESH_DIR="/work/m24046/m24046mrcr/results_data_30min_35_70/"
COARSE_MESH_DIR="/work/m24046/m24046mrcr/results_data_30min_35_70_maillagex${NUMBER}/"

# Specify the list of .res files to process (space-separated list)
RES_FILES=(
    "Group_3_peak_1000_Group_3_peak_1000_0_0-80.pkl"
    "Group_3_peak_1200_Group_3_peak_1200_0_0-80.pkl"
    "Group_3_peak_1400_Group_3_peak_1400_0_0-80.pkl"
    "Group_3_peak_1600_Group_3_peak_1600_0_0-80.pkl"
    "Group_3_peak_1800_Group_3_peak_1800_0_0-80.pkl"
    "Group_3_peak_2000_Group_3_peak_2000_0_0-80.pkl"
    "Group_3_peak_2200_Group_3_peak_2200_0_0-80.pkl"
    "Group_3_peak_2400_Group_3_peak_2400_0_0-80.pkl"
    "Group_3_peak_2600_Group_3_peak_2600_0_0-80.pkl"
    "Group_3_peak_2800_Group_3_peak_2800_0_0-80.pkl"
    "Group_3_peak_3000_Group_3_peak_3000_0_0-80.pkl"
    "Group_3_peak_3200_Group_3_peak_3200_0_0-80.pkl"
    "Group_3_peak_3400_Group_3_peak_3400_0_0-80.pkl"
    "Group_3_peak_3600_Group_3_peak_3600_0_0-80.pkl"
    # Add more files as needed
)

# Path to the interpolate_mesh.py script
INTERPOLATE_SCRIPT="/users/m24046/m24046mrcr/gnn_modulus_test/bin/create_interpolate_dataset.py"

# Build the command to run the interpolate_mesh.py script
# We use "${RES_FILES[@]}" to pass all filenames as separate arguments
apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif python create_interpolate_dataset.py "$FINE_MESH_DIR" "$COARSE_MESH_DIR" --pkl_files "${RES_FILES[@]}"

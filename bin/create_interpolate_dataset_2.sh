#!/bin/sh
#SBATCH --job-name=interpolate
#SBATCH --output=interpolate-%j.out
#SBATCH --error=interpolate-%j.err
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=gpu

NUMBER=${1:-8}   # keep your CLI arg; default to 8 if omitted

FINE_MESH_DIR="/work/m24046/m24046mrcr/results_data_30min_35_70/"
COARSE_MESH_DIR="/work/m24046/m24046mrcr/results_data_30min_35_70_maillagex${NUMBER}/"

RES_FILES=(
  "Group_2_peak_1000_Group_2_peak_1000_0_0-80.pkl"
  "Group_2_peak_1000_Group_2_peak_1000_0_0-80.pkl"
  "Group_2_peak_1200_Group_2_peak_1200_0_0-80.pkl"
  "Group_2_peak_1400_Group_2_peak_1400_0_0-80.pkl"
  "Group_2_peak_1600_Group_2_peak_1600_0_0-80.pkl"
  "Group_2_peak_1800_Group_2_peak_1800_0_0-80.pkl"
  "Group_2_peak_2000_Group_2_peak_2000_0_0-80.pkl"
  "Group_2_peak_2200_Group_2_peak_2200_0_0-80.pkl"
  "Group_2_peak_2400_Group_2_peak_2400_0_0-80.pkl"
  "Group_2_peak_2600_Group_2_peak_2600_0_0-80.pkl"
  "Group_2_peak_2800_Group_2_peak_2800_0_0-80.pkl"
  "Group_2_peak_3000_Group_2_peak_3000_0_0-80.pkl"
  "Group_2_peak_3200_Group_2_peak_3200_0_0-80.pkl"
  "Group_2_peak_3400_Group_2_peak_3400_0_0-80.pkl"
  "Group_2_peak_3600_Group_2_peak_3600_0_0-80.pkl"
)

INTERPOLATE_SCRIPT="/users/m24046/m24046mrcr/gnn_modulus_test/bin/create_interpolate_dataset.py"

# Ensure both /users and /work are visible inside the container
for f in "${RES_FILES[@]}"; do
  apptainer exec --bind /tmpdir,/work,/users --nv \
    /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
    python "$INTERPOLATE_SCRIPT" "$FINE_MESH_DIR" "$COARSE_MESH_DIR" --pkl_files "$f"
done


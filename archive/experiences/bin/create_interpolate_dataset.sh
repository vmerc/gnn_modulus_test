#!/bin/sh
#SBATCH --job-name=interpolate
#SBATCH --output=interpolate-%j.out
#SBATCH --error=interpolate-%j.err
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00


FINE_MESH_DIR="/work/m24046/m24046mrcr/dataset_x8_avec_ts/full/"
COARSE_MESH_DIR="/work/m24046/m24046mrcr/dataset_x8_avec_ts/fullx8/"
RES_FILES=( 
    "Group_3_peak_2800_Group_3_peak_2800_0_0-80.pkl"
    "Group_4_peak_1200_Group_4_peak_1200_0_0-80.pkl"
    "Group_4_peak_3000_Group_4_peak_3000_0_0-80.pkl"
)

INTERPOLATE_SCRIPT="/users/m24046/m24046mrcr/gnn_modulus_test/bin/create_interpolate_dataset_with_ts.py"

# Ensure both /users and /work are visible inside the container
for f in "${RES_FILES[@]}"; do
  apptainer exec --bind /tmpdir,/work,/users --nv \
    /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
    python "$INTERPOLATE_SCRIPT" "$FINE_MESH_DIR" "$COARSE_MESH_DIR" --pkl_files "$f"
done


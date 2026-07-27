#!/bin/sh
#SBATCH --job-name=create-multimesh
#SBATCH --output=create_multimesh-%j.out
#SBATCH --error=create_multimesh-%j.err
#SBATCH -p shared
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:0

if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch create_multimesh.sh --base-graph BASE.bin --fine-mesh FINE.slf --coarse-mesh COARSE.slf [COARSE.slf ...] --output MULTIMESH.bin"
  exit 1
fi


srun apptainer exec --bind /tmpdir,/work,/users --nv \
  /work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif \
  python "create_multimesh.py" "$@"

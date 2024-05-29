#!/bin/sh

#SBATCH --job-name=GPU-GNN-test
#SBATCH --output=ML-%j-gnn.out
#SBATCH --error=ML-%j-gnn.err

#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding

module purge
module load singularity/3.0.3

srun singularity exec /logiciels/containerCollections/CUDA12/pytorch2-NGC-23-05-py3.sif $HOME/env_dgl/bin/python "train_script.py"
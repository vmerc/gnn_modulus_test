#!/bin/bash
#SBATCH --job-name=test_shared
#SBATCH --output=test_shared-%j.out
#SBATCH --error=test_shared-%j.err
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=40          # 40 tasks max on shared
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1         # 1 GPU max on shared
#SBATCH --time=00:10:00      # short run to test
#SBATCH --ntasks-per-node=40

echo "Hello from shared partition"
echo "JobID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
nvidia-smi               # check GPU visible
hostname
sleep 300                 # hold for 5 minutes

#!/bin/bash

#SBATCH --job-name=ppo_training
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=20:00:00
#SBATCH --output=sbatch_logs/ppo_training_%j.out
#SBATCH --error=sbatch_logs/ppo_training_%j.err

echo "Running on $(hostname)"
echo "Starting job at: $(date)"

module load anaconda3/2022.10/gcc-13.1.0
module load cuda/12.1.1/gcc-13.1.0

source activate rl_env

python highway/models/ppo_sb3/train.py
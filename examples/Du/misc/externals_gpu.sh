#!/bin/bash
#SBATCH --job-name=find_params
#SBATCH --array=1-12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%a.%j.out
#SBATCH --error=logs/%a.%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistics

cd $PWD

python -u externals.py $SLURM_ARRAY_TASK_ID
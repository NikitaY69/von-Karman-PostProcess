#!/bin/bash
#SBATCH --job-name=find_params
#SBATCH --array=1-12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=100GB
#SBATCH --output=logs/f.%a.out
#SBATCH --error=logs/f.%a.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistics

cd $PWD

python -u externals.py $SLURM_ARRAY_TASK_ID
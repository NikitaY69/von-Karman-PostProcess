#!/bin/bash
#SBATCH --job-name=joint_pdf
#SBATCH --array=1-8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150GB
#SBATCH --output=logs/%a.%j.out
#SBATCH --error=logs/%a.%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistics

cd $PWD

python -u joint_pdf.py $SLURM_ARRAY_TASK_ID
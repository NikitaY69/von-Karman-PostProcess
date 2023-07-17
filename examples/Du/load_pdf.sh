#!/bin/bash
#SBATCH --job-name=plot_pdf
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-16
#SBATCH --mem=50GB
#SBATCH --output=figs/logs/%a.%j.out
#SBATCH --error=figs/logs/%a.%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistics

cd $PWD

python load_pdf.py $SLURM_ARRAY_TASK_ID
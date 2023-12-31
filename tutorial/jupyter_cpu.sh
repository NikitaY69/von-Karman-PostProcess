#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=10GB
#SBATCH --output=out_cpu
#SBATCH --error=log_cpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistics

cd $PWD
jupyter notebook --no-browser --port=8890 --ip=0.0.0.0
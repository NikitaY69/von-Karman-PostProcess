#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --exclude=n51,n52,n53,n54,n55,n101,n102
#SBATCH --mem=100GB
#SBATCH --output=out_gpu
#SBATCH --error=log_gpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistics

cd $PWD
jupyter notebook --no-browser --port=8889 --ip=0.0.0.0

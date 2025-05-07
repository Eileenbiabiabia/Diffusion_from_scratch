#!/bin/sh
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --time=01:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=output_logs/slurm-%j.out

module purge
eval "$(conda shell.bash hook)"
conda activate genai

nvidia-smi
echo "Starting the preprocessing script..."
python preprocess.py
echo "preprocess script completed. Please check the output logs for details."
#!/bin/sh
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=15:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --output=output_logs/slurm-%j.out


module purge
#eval "$(conda shell.bash hook)"
source /home/azi8380/miniconda3/etc/profile.d/conda.sh
conda activate genai

echo "Checking GPU memory status before starting..."
nvidia-smi
echo "Starting the training script..."
python train.py
echo "Training script completed."
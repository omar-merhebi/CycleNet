#!/bin/bash
#SBATCH -J train_no_augments
#SBATCH -t 04:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/R-%x.%j.out
#SBATCH --error=jobs/R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source ~/.bashrc

pyenv activate PHASER

module load gcc
module load cuda

python main.py dataset.augment=false +wandb.tags=[all_channels,predict_phase,masking,no_augments,no_fill] wandb.run_name=no_augments
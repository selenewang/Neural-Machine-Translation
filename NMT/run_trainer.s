#!/bin/bash
#
#SBATCH --job-name=train_nmt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=50:59:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1

python trainer.py en zh data

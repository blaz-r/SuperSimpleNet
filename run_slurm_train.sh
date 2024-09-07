#!/bin/sh
#SBATCH --job-name=ssn_icpr
#SBATCH --output=./sout/ssn_icpr%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=50:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu

srun --kill-on-bad-exit=1 python -u train.py mvtec
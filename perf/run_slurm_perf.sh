#!/bin/sh
#SBATCH --job-name=perf
#SBATCH --output=./sout/perf_%j_%N.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --partition=gpu
#SBATCH --exclusive

srun --kill-on-bad-exit=1 python -u perf_main.py v100s
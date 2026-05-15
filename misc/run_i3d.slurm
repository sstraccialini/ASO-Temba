#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --job-name=i3d_extract
#SBATCH --output=i3d_extract_%j.out
#SBATCH --error=i3d_extract_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

cd ~/pytorch-i3d

eval "$(conda shell.bash hook)"
conda activate cv_project

python pytorch-i3d/extract_features.py\
  -mode rgb \
  -load_model pytorch-i3d/models/rgb_charades.pt \
  -root RGB/Videos_frames \
  -save_dir data/TSU/i3d_features \
  -frames_per_segment 16 \
  -target_T 2500 \
  -pad_mode zero \
  -save_format npy
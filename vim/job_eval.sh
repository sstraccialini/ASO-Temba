#!/bin/bash
#SBATCH --account=3185670
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-eval-%j.out

source ~/.bashrc
conda activate cv_project
cd /mnt/beegfsstudents/home/3185670/ASO-Temba/vim

export PYTHONPATH=/home/3185670/ASO-Temba/mamba-1p1p1:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python MSTemba_main.py \
  -dataset tsu \
  -mode rgb \
  -train eval \
  -backbone i3d \
  -gpu 0 \
  -batch_size 4 \
  -num_clips 8 \
  -skip 1 \
  -rgb_root /home/3185670/i3d_features \
  -load_model /home/3185670/ASO-Temba/outputs/tsu_i3d/best_model.pth \
  -output_dir /home/3185670/ASO-Temba/outputs/tsu_i3d_eval

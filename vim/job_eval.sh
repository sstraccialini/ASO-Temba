#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID
USER_HOME=/home/$MYID

# ====== ENV SETUP ======
source ~/.bashrc
conda activate cv_project

cd $BASE_HOME/ASO-Temba/vim

export PYTHONPATH=$BASE_HOME/ASO-Temba/mamba-1p1p1:$PYTHONPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ====== DEBUG INFO ======
hostname
which python
echo "MYID=$MYID"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python -m pip show mamba-ssm || true
echo "======================"

python MSTemba_main.py \
  -dataset tsu \
  -mode rgb \
  -train eval \
  -backbone i3d \
  -gpu 0 \
  -batch_size 1 \
  -num_clips 2500 \
  --opt adamw \
  --lr 4.5e-4 \
  --sched cosine \
  --warmup-epochs 5 \
  -epochs 140 \
  --model mstemba \
-skip 1 \
-rgb_root $BASE_HOME/ASO-Temba/data/TSU/i3d_features \
-load_model $BASE_HOME/ASO-Temba/outputs/tsu_i3d/best_model.pth \
-output_dir $BASE_HOME/ASO-Temba/outputs/tsu_i3d_eval

echo "Evaluation done."

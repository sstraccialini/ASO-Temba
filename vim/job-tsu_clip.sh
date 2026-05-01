#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-tsu_clip-%j.out
#SBATCH --cpus-per-task=8

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID
USER_HOME=/home/$MYID

# ====== ENV SETUP ======
source ~/.bashrc
conda activate mstemba

cd $BASE_HOME/ASO-Temba/vim

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

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
  -backbone clip \
  -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/tsu_features_clip_l14 \
  -num_clips 2500 \
  -skip 0 \
  --lr 4.5e-4 \
  -comp_info False \
  -epochs 140 \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0.05 \
  -batch_size 1 \
  --num_workers 1 \
  -output_dir $BASE_HOME/ASO-Temba/outputs/tsu_clip

echo "Training done."

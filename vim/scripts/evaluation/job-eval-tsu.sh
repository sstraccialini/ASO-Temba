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

echo "Evaluation..."

echo "chunk_size=1"

python streaming_inference.py \
  --weights $BASE_HOME/ASO-Temba/outputs/causal-tsu/best_model.pth \
  --dataset tsu \
  --backbone clip/i3d \
  --rgb_root $BASE_HOME/ASO-Temba/data/tsu_features_i3d or $BASE_HOME/ASO-Temba/data/tsu_features_clip_l14 \
  --stream_chunk_size 1 \
  --streaming_demo_n 50

echo "chunk_size=25"

python streaming_inference.py \
  --weights $BASE_HOME/ASO-Temba/outputs/causal-tsu/best_model.pth \
  --dataset tsu \
  --backbone i3d/clip \
  --rgb_root $BASE_HOME/ASO-Temba/data/tsu_features_i3d or $BASE_HOME/ASO-Temba/data/tsu_features_clip_l14 \
  --stream_chunk_size 25 \
  --streaming_demo_n 50


echo "Evaluation done."

#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-demo-online-%j.out
#SBATCH --cpus-per-task=4

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID

# ====== ENV SETUP ======
source ~/.bashrc
conda activate mstemba

cd $BASE_HOME/ASO-Temba

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

hostname
which python
echo "MYID=$MYID"

# ====== ONLINE INFERENCE DEMO ======
# Generates an MP4 showing a random test video with live predicted labels.
# Saved to visualization/demo_tsu.mp4 (no display needed on headless node).

python visualization/demo_online.py \
  --weights $BASE_HOME/ASO-Temba/outputs/causal-tsu_i3d/best_model.pth \
  --dataset tsu \
  --backbone i3d \
  --rgb_root $BASE_HOME/ASO-Temba/data/tsu_features_i3d \
  --step 1 \
  --interval 60 \
  --save visualization/demo_tsu.mp4


# ---- Charades variant (uncomment to use) ----
# python visualization/demo_online.py \
#   --weights $BASE_HOME/ASO-Temba/outputs/causal-char_i3d-token_attention/best_model.pth \
#   --dataset charades \
#   --backbone i3d \
#   --rgb_root $BASE_HOME/ASO-Temba/data/charades_features_i3d \
#   --fuser token-attention \
#   --step 1 \
#   --interval 60 \
#   --save visualization/demo_charades.mp4

echo "Demo MP4 saved."

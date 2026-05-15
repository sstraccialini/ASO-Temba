#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-causal-tsu_i3d-%j.out
#SBATCH --cpus-per-task=8

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID

# ====== ENV SETUP ======
source ~/.bashrc
conda activate mstemba

cd $BASE_HOME/ASO-Temba/vim

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

hostname
which python
echo "MYID=$MYID"

# ====== CAUSAL TRAINING (TSU / I3D) ======
# Key differences vs offline:
#   --causal                         enables forward-only SSMs
#   --causal_consistency_loss_weight adds L_caus_cons regularisation
#   --causal_consistency_margin      hinge margin for L_caus_cons


python MSTemba_main.py \
  -dataset charades \
  -mode rgb \
  -backbone i3d/clip \
  -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/charades_features_i3d or $BASE_HOME/ASO-Temba/data/charades_features_clip \
  -num_clips 256 \
  -skip 0 \
  --lr 2e-4 \
  -comp_info False \
  -epochs 100 \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0.05 \
  -batch_size 5 \
  --num_workers 0 \
  --causal \
  --causal_consistency_loss_weight 1.0 \
  --causal_consistency_margin 0.1 \
  --drop 0.15 \
  --drop-path 0.15 \
  --weight-decay 0.05 \
  --clip-grad 1.0 \
  --warmup-epochs 10 \
  --min-lr 1e-5 \
  --fuser sum/token-attention \
  -output_dir $BASE_HOME/ASO-Temba/outputs/causal-charades


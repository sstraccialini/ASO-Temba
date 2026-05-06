#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-attx3_all-%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ====== USER CONFIG ======
MYID=3194064
BASE_HOME=/mnt/beegfsstudents/home/$MYID
OUTPUT_BASE=$BASE_HOME/ASO-Temba/outputs/attx3

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
python -m pip show mamba-ssm || true
echo "======================"

mkdir -p $OUTPUT_BASE/tsu_i3d
mkdir -p $OUTPUT_BASE/tsu_clip
mkdir -p $OUTPUT_BASE/char_i3d
mkdir -p $OUTPUT_BASE/char_clip

# --- RUN 1: TSU I3D (250 epoche) ---
python MSTemba_main.py \
  -dataset tsu -mode rgb -backbone i3d -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/tsu_features_i3d \
  -num_clips 2500 -skip 0 -comp_info False \
  -epochs 250 -unisize True \
  -alpha_l 1.0 -beta_l 0.05 \
  -batch_size 1 --num_workers 1 \
  --fuser attention_x3 \
  --lr 0.00045 --weight-decay 0.05 \
  --drop-path 0.1 --drop 0.0 \
  --warmup-epochs 10 --warmup-lr 1e-6 --min-lr 1e-5 \
  --clip-grad 1.0 \
  --diversity-loss-weight 0.1 --block-loss-weight 0.02 \
  --model-ema --model-ema-decay 0.99996 \
  -output_dir $OUTPUT_BASE/tsu_i3d \
  > $OUTPUT_BASE/tsu_i3d/training.log 2>&1
echo "TSU I3D done."

# --- RUN 2: TSU CLIP (250 epoche) ---
python MSTemba_main.py \
  -dataset tsu -mode rgb -backbone clip -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/tsu_features_clip \
  -num_clips 2500 -skip 0 -comp_info False \
  -epochs 250 -unisize True \
  -alpha_l 1.0 -beta_l 0.05 \
  -batch_size 1 --num_workers 1 \
  --fuser attention_x3 \
  --lr 0.00045 --weight-decay 0.05 \
  --drop-path 0.1 --drop 0.0 \
  --warmup-epochs 10 --warmup-lr 1e-6 --min-lr 1e-5 \
  --clip-grad 1.0 \
  --diversity-loss-weight 0.1 --block-loss-weight 0.02 \
  --model-ema --model-ema-decay 0.99996 \
  -output_dir $OUTPUT_BASE/tsu_clip \
  > $OUTPUT_BASE/tsu_clip/training.log 2>&1
echo "TSU CLIP done."

# --- RUN 3: Charades I3D (120 epoche) ---
python MSTemba_main.py \
  -dataset charades -mode rgb -backbone i3d -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/charades_features_i3d \
  -num_clips 256 -skip 0 -comp_info False \
  -epochs 120 -unisize True \
  -alpha_l 1 -beta_l 0.05 \
  -batch_size 5 --num_workers 0 \
  --fuser attention_x3 \
  --lr 0.00045 --weight-decay 0.05 \
  --drop-path 0.1 --drop 0.0 \
  --warmup-epochs 10 --warmup-lr 1e-6 --min-lr 1e-5 \
  --clip-grad 1.0 \
  --diversity-loss-weight 0.1 --block-loss-weight 0.02 \
  --model-ema --model-ema-decay 0.99996 \
  -output_dir $OUTPUT_BASE/char_i3d \
  > $OUTPUT_BASE/char_i3d/training.log 2>&1
echo "Charades I3D done."

# --- RUN 4: Charades CLIP (120 epoche) ---
python MSTemba_main.py \
  -dataset charades -mode rgb -backbone clip -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/charades_features_clip \
  -num_clips 256 -skip 0 -comp_info False \
  -epochs 120 -unisize True \
  -alpha_l 1 -beta_l 0.05 \
  -batch_size 5 --num_workers 0 \
  --fuser attention_x3 \
  --lr 0.00045 --weight-decay 0.05 \
  --drop-path 0.1 --drop 0.0 \
  --warmup-epochs 10 --warmup-lr 1e-6 --min-lr 1e-5 \
  --clip-grad 1.0 \
  --diversity-loss-weight 0.1 --block-loss-weight 0.02 \
  --model-ema --model-ema-decay 0.99996 \
  -output_dir $OUTPUT_BASE/char_clip \
  > $OUTPUT_BASE/char_clip/training.log 2>&1
echo "Charades CLIP done."

echo "ALL RUNS COMPLETED."

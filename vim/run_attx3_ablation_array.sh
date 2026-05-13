#!/bin/bash
#SBATCH --job-name=attx3_ablation
#SBATCH --output=/mnt/beegfsstudents/home/3194064/ASO-Temba/slurm_logs/attx3_ablation_%A_%a.out
#SBATCH --error=/mnt/beegfsstudents/home/3194064/ASO-Temba/slurm_logs/attx3_ablation_%A_%a.err
#SBATCH --array=0-15%1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ---------------------------------------------------------------------------
# AttentionX3 ablation array job
# 4 fusers × 4 dataset/backbone setups = 16 tasks
# ---------------------------------------------------------------------------

MYID=3194064
BASE_HOME=/mnt/beegfsstudents/home/$MYID
REPO=$BASE_HOME/ASO-Temba/vim
OUTPUT_BASE=$BASE_HOME/ASO-Temba/outputs/attx3_ablations

# ---- task mapping ---------------------------------------------------------
FUSERS=(
    attention_x3_no_attn
    attention_x3_no_ffn
    attention_x3_bn
    attention_x3_shared_common
)

SETUPS=(
    tsu_i3d
    tsu_clip
    char_i3d
    char_clip
)

# SLURM_ARRAY_TASK_ID = fuser_idx * 4 + setup_idx
FUSER_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
SETUP_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

FUSER=${FUSERS[$FUSER_IDX]}
SETUP=${SETUPS[$SETUP_IDX]}

# ---- environment ----------------------------------------------------------
source ~/.bashrc
conda activate mstemba
cd $REPO
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ---- per-setup hyperparameters --------------------------------------------
case $SETUP in
    tsu_i3d)
        DATASET=tsu
        BACKBONE=i3d
        RGB_ROOT=$BASE_HOME/ASO-Temba/data/tsu_features_i3d
        NUM_CLIPS=2500
        BATCH_SIZE=1
        NUM_WORKERS=1
        EPOCHS=250
        LR=0.0003
        WEIGHT_DECAY=0.07
        DROP=0.0
        DROP_PATH=0.15
        WARMUP_EPOCHS=15
        ;;
    tsu_clip)
        DATASET=tsu
        BACKBONE=clip
        RGB_ROOT=$BASE_HOME/ASO-Temba/data/tsu_features_clip_l14
        NUM_CLIPS=2500
        BATCH_SIZE=1
        NUM_WORKERS=1
        EPOCHS=250
        LR=0.00035
        WEIGHT_DECAY=0.05
        DROP=0.0
        DROP_PATH=0.1
        WARMUP_EPOCHS=10
        ;;
    char_i3d)
        DATASET=charades
        BACKBONE=i3d
        RGB_ROOT=$BASE_HOME/ASO-Temba/data/charades_features_i3d
        NUM_CLIPS=256
        BATCH_SIZE=5
        NUM_WORKERS=0
        EPOCHS=50
        LR=0.00025
        WEIGHT_DECAY=0.08
        DROP=0.1
        DROP_PATH=0.2
        WARMUP_EPOCHS=15
        ;;
    char_clip)
        DATASET=charades
        BACKBONE=clip
        RGB_ROOT=$BASE_HOME/ASO-Temba/data/charades_features_clip
        NUM_CLIPS=256
        BATCH_SIZE=5
        NUM_WORKERS=0
        EPOCHS=50
        LR=0.0003
        WEIGHT_DECAY=0.06
        DROP=0.05
        DROP_PATH=0.15
        WARMUP_EPOCHS=15
        ;;
esac

RUN_NAME=${FUSER}_${SETUP}
OUTPUT_DIR=$OUTPUT_BASE/$FUSER/$RUN_NAME

# ---- diagnostics ----------------------------------------------------------
echo "hostname:     $(hostname)"
echo "which python: $(which python)"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "FUSER:        $FUSER"
echo "RUN_NAME:     $RUN_NAME"
echo "OUTPUT_DIR:   $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR
mkdir -p $BASE_HOME/ASO-Temba/slurm_logs

# ---- training -------------------------------------------------------------
python MSTemba_main.py \
    -dataset $DATASET \
    -backbone $BACKBONE \
    -rgb_root $RGB_ROOT \
    -num_clips $NUM_CLIPS \
    -batch_size $BATCH_SIZE \
    -num_workers $NUM_WORKERS \
    -epochs $EPOCHS \
    -lr $LR \
    -weight_decay $WEIGHT_DECAY \
    -drop $DROP \
    -drop_path $DROP_PATH \
    -warmup_epochs $WARMUP_EPOCHS \
    -warmup_lr 1e-6 \
    -min_lr 1e-5 \
    -clip_grad 1.0 \
    -diversity_loss_weight 0.05 \
    -block_loss_weight 0.01 \
    -alpha_l 1.0 \
    -beta_l 0.05 \
    --model_ema True \
    --model_ema_decay 0.99996 \
    --fuser $FUSER \
    -output_dir $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/training.log

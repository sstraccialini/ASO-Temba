#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --job-name=serial_extract
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID
USER_HOME=/home/$MYID

# ---------- Settings ----------
VIDEO_DIR="$BASE_HOME/datasets/RGB/Videos_mp4"
FRAME_DIR="$BASE_HOME/datasets/RGB/Videos_frames"
FEAT_DIR="$BASE_HOME/ASO-Temba/data/TSU/i3d_features"
FPS=24
ENV_NAME="extract_features"
MODEL_PATH="pytorch-i3d/models/rgb_charades.pt"
# ------------------------------

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd $BASE_HOME/ASO-Temba

mkdir -p "$FRAME_DIR"
mkdir -p "$FEAT_DIR"

for video in "$VIDEO_DIR"/*.mp4; do
    base=$(basename "$video")
    name="${base%.mp4}"

    # Create an isolated root for this specific video so extract_features.py doesn't see others
    TMP_ROOT="$FRAME_DIR/batch_$name"
    outdir="$TMP_ROOT/$name"

    # Skip if features already exist
    if [ -f "$FEAT_DIR/$name.npy" ]; then
        echo "Features for $name already exist. Skipping."
        continue
    fi

    echo "--- Processing $name ---"

    # 1. Extract frames into the isolated folder
    mkdir -p "$outdir"
    ffmpeg -hide_banner -loglevel error -y -i "$video" -vf "fps=$FPS" -q:v 2 "$outdir/frame_%04d.jpg"

    # 2. Extract features (using target_T 0, batch scaling, and handling deletion safely in bash)
    python pytorch-i3d/extract_features.py \
        -mode rgb \
        -load_model "$MODEL_PATH" \
        -root "$TMP_ROOT" \
        -save_dir "$FEAT_DIR" \
        -frames_per_segment 16 \
        -batch_size 32 \
        -target_T 0 \
        -pad_mode zero

    # 3. Clean up ONLY if feature extraction succeeded
    if [ -f "$FEAT_DIR/$name.npy" ]; then
        echo "[Cleanup] Successfully generated features. Cleaning up $name..."
        rm -rf "$TMP_ROOT"
        rm -f "$video"
    else
        echo "[Error] Feature file not generated. Keeping frames and mp4 for debugging."
    fi

    echo "--- Finished $name ---"
done

echo "Pipeline complete."
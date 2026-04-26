#!/bin/bash
# SBATCH --job-name=serial_pipeline
# SBATCH --output=serial_pipeline_%j.out
# SBATCH --error=serial_pipeline_%j.err
# SBATCH --time=24:00:00
# SBATCH --cpus-per-task=4
# SBATCH --mem=64G
# SBATCH --gres=gpu:1

set -euo pipefail

# ---------- Settings ----------
VIDEO_DIR="RGB/Videos_mp4"
FRAME_DIR="RGB/Videos_frames"
FEAT_DIR="data/TSU/i3d_features"
FPS=24
ENV_NAME="cv_project"
MODEL_PATH="pytorch-i3d/models/rgb_charades.pt"
# ------------------------------

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

mkdir -p "$FRAME_DIR"
mkdir -p "$FEAT_DIR"

for video in "$VIDEO_DIR"/*.mp4; do
    base=$(basename "$video")
    name="${base%.mp4}"
    outdir="$FRAME_DIR/$name"
    
    # Skip if features already exist
    if [ -f "$FEAT_DIR/$name.npy" ]; then
        echo "Features for $name already exist. Skipping."
        continue
    fi

    echo "--- Processing $name ---"
    
    # 1. Extract frames
    mkdir -p "$outdir"
    ffmpeg -hide_banner -loglevel error -y -i "$video" -vf "fps=$FPS" "$outdir/frame_%04d.jpg"
    
    # 2. Extract features (using updated script with auto-delete)
    python pytorch-i3d/extract_features.py \
        -mode rgb \
        -load_model "$MODEL_PATH" \
        -root "$FRAME_DIR" \
        -save_dir "$FEAT_DIR" \
        -frames_per_segment 16 \
        -target_T 2500 \
        -pad_mode zero \
        -delete_frames \
        -delete_video \
        -video_root "$VIDEO_DIR"

    echo "--- Finished $name ---"
done

echo "Pipeline complete."

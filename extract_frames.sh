#!/bin/bash
#SBATCH --job-name=extract_frames
#SBATCH --output=extract_frames_%j.out
#SBATCH --error=extract_frames_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# ---------- User settings ----------
ENV_NAME="cv_project"
INPUT_DIR="${1:-$HOME/Videos_mp4}"
OUTPUT_DIR="${2:-$HOME/Videos_frames}"
FPS="${3:-5}"
EXT="${4:-jpg}"   # jpg or png
# ----------------------------------

echo "Input dir : $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "FPS       : $FPS"
echo "Format    : $EXT"
echo "CPUs      : ${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$OUTPUT_DIR"

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
else
    echo "ERROR: conda command not found"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found in environment '$ENV_NAME'"
    exit 1
fi

shopt -s nullglob
mapfile -t files < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.mp4' | sort)

if [ "${#files[@]}" -eq 0 ]; then
    echo "No .mp4 files found in $INPUT_DIR"
    exit 1
fi

export OUTPUT_DIR FPS EXT

process_one() {
    local f="$1"
    local base name outdir
    base="$(basename "$f")"
    name="${base%.mp4}"
    outdir="$OUTPUT_DIR/$name"
    mkdir -p "$outdir"

    echo "Processing $base -> $outdir"
    ffmpeg -hide_banner -loglevel error -y \
        -i "$f" \
        -vf "fps=$FPS" \
        "$outdir/frame_%04d.$EXT"
}
export -f process_one

# Run several videos in parallel inside one Slurm job
printf '%s\n' "${files[@]}" | xargs -I {} -n 1 -P "${SLURM_CPUS_PER_TASK:-1}" bash -c 'process_one "$@"' _ {}

echo "Done."

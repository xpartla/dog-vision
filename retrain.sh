#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="dogvision"
PYTHON="conda run -n $CONDA_ENV python"
ROOT="$(cd "$(dirname "$0")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$ROOT"

log "=== Step 1/3: Running pose estimation on new clips ==="

NEW_SITTING=(
    data/sitting/sitting-11.mp4
    data/sitting/sitting-12.mp4
    data/sitting/sitting-13.mp4
    data/sitting/sitting-14.mp4
    data/sitting/sitting-15.mp4
)
NEW_STANDING=(
    data/standing/standing-10.mp4
    data/standing/standing-11.mp4
    data/standing/standing-12.mp4
    data/standing/standing-13.mp4
    data/standing/standing-14.mp4
    data/standing/standing-15.mp4
    data/standing/standing-16.mp4
    data/standing/standing-17.mp4
    data/standing/standing-18.mp4
)
NEW_LYING=(
    data/lying/laying-11.mp4
    data/lying/laying-12.mp4
    data/lying/laying-13.mp4
    data/lying/laying-14.mp4
    data/lying/laying-15.mp4
)

process_clip() {
    local video="$1"
    local out_dir
    out_dir="$(dirname "$video")"
    local stem
    stem="$(basename "$video" .mp4)"
    local h5_glob="${out_dir}/${stem}_superanimal_quadruped"*".h5"

    if ls $h5_glob 2>/dev/null | grep -q .; then
        log "  SKIP $video (h5 already exists)"
        return
    fi

    log "  Processing $video ..."
    $PYTHON process_video.py "$video" --output-dir "$out_dir"
    log "  Done: $video"
}

TOTAL=$(( ${#NEW_SITTING[@]} + ${#NEW_STANDING[@]} + ${#NEW_LYING[@]} ))
DONE=0

for v in "${NEW_SITTING[@]}" "${NEW_STANDING[@]}" "${NEW_LYING[@]}"; do
    process_clip "$v"
    DONE=$(( DONE + 1 ))
    log "  Progress: $DONE/$TOTAL clips"
done

log "=== Step 2/3: Rebuilding dataset ==="
$PYTHON build_dataset.py data/ --out dataset.npz --stride 2 --augment-flip

log "=== Step 3/3: Retraining models (RF + MLP) ==="
$PYTHON train_posture.py dataset.npz --model rf  --out posture_model.joblib
$PYTHON train_posture.py dataset.npz --model mlp --out posture_model_mlp.joblib

log "=== All done. RF → posture_model.joblib, MLP → posture_model_mlp.joblib ==="

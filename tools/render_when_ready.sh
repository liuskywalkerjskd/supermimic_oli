#!/bin/bash
# Wait for a checkpoint, record all evaluation rollouts, and render a highlight video.
# Usage: tools/render_when_ready.sh <run_dir> <iteration> <motion_dir> <gpu> [clips...]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=$1; IT=$2; MOTION=$3; GPU=$4; shift 4
CLIPS=${*:-dance2_subject1 jumps1_subject1 fallAndGetUp1_subject1 sprint1_subject2 walk1_subject2 fight1_subject2}
CK=$(printf "%s/ckpt_%06d.pt" "$RUN" "$IT")
PY=${PYTHON:-python}
export SM_ROBOT=oli __EGL_VENDOR_LIBRARY_FILENAMES=$PWD/docker/10_nvidia.json
while [ ! -f "$CK" ]; do sleep 60; done
sleep 30  # checkpoint is written via rename; give the eval watcher a head start
TRAJ=$RUN/traj_$IT.npz
CUDA_VISIBLE_DEVICES=$GPU $PY supermimic/evaluate.py --run_dir "$RUN" --motion_dir "$MOTION" --ckpt "$CK" --record "$TRAJ" > "$RUN/record_$IT.json.log" 2>&1
TITLE="Oli $(basename "$RUN") it $IT"
$PY tools/render.py "$TRAJ" "$RUN/oli_it${IT}_highlights.mp4" --clips $CLIPS --seconds 10 --title "$TITLE"
$PY tools/render.py "$TRAJ" "$RUN/oli_it${IT}_all_clips.mp4" --seconds 6 --start 2 --title "$TITLE"
echo "RENDER_DONE $RUN $IT"

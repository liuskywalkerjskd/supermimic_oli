#!/bin/bash
# Launch SuperMimic training in a tmux session inside the 'supermimic' container.
# Usage: [CPUS=0-23] ./run_train.sh <run_name> [train.py args...]
# CPUS optionally pins the run with taskset (rootless docker ignores --cpuset-cpus).
RUN=${1:-lafan1_main}
shift || true
PIN=""
[ -n "${CPUS:-}" ] && PIN="taskset -c $CPUS"
docker exec supermimic tmux new-session -d -s "$RUN" -e "SM_ROBOT=${SM_ROBOT:-oli}" \
  "cd /workspace/supermimic && mkdir -p logs/$RUN && $PIN python supermimic/train.py --log_dir logs/$RUN $* 2>&1 | tee -a logs/$RUN/stdout.log"

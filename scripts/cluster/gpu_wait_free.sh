#!/usr/bin/env bash
# Block until the GPU selected by CUDA_VISIBLE_DEVICES (first index) has at least <min_free_mib> free, or give up.
#
#   bash gpu_wait_free.sh <min_free_mib> [max_wait_minutes=60]      -> exit 0 when free enough, 1 on timeout
#
# Used between queue stages: a job killed from outside can leave a zombie holding tens of GB on the card, and a
# co-tenant's job may come back; starting the next stage blindly then fails with CUDA OOM.
set -uo pipefail
MIN_FREE="${1:?min free MiB}"; MAX_MIN="${2:-60}"
IDX="${CUDA_VISIBLE_DEVICES%%,*}"; IDX="${IDX:-0}"
deadline=$(( $(date +%s) + MAX_MIN * 60 ))
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$IDX" 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "${free:-}" ] && [ "$free" -ge "$MIN_FREE" ]; then
    echo "[gpu_wait] $(date '+%F %T') GPU $IDX has ${free} MiB free (>= $MIN_FREE)"; exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[gpu_wait] $(date '+%F %T') GPU $IDX still only ${free:-?} MiB free after $MAX_MIN min; giving up"; exit 1
  fi
  sleep 60
done

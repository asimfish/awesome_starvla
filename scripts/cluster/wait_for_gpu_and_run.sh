#!/usr/bin/env bash
# Wait for a GPU with enough free memory on this node, take a shared-filesystem lock, and run a command on that GPU.
# Meant to be started (nohup) on every node that shares $WORK; whichever node finds a free card first wins the lock.
#
#   bash wait_for_gpu_and_run.sh <min_free_mib> <lock_name> [max_wait_minutes] -- <command...>
#   e.g. bash wait_for_gpu_and_run.sh 70000 f4 720 -- bash scripts/cluster/run_f4_fp32.sh
#
# The command runs with CUDA_VISIBLE_DEVICES set to the chosen index. The lock is $WORK/locks/<lock_name> (mkdir is
# atomic on the shared mount) and is not removed on exit, so a finished job is never re-run; delete it to re-arm.
# While the command runs, $WORK/locks/gpu_<host>_<idx> marks the card so other waiters on the host pick a different one.
set -uo pipefail

MIN_FREE="${1:?min free MiB}"; LOCK_NAME="${2:?lock name}"; shift 2
MAX_WAIT_MIN=720
if [ "${1:-}" != "--" ]; then MAX_WAIT_MIN="$1"; shift; fi
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "no command given"; exit 2; }

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
LOCK="$WORK/locks/$LOCK_NAME"
mkdir -p "$WORK/locks"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))

while :; do
  if [ -d "$LOCK" ]; then echo "[wait] $(date '+%F %T') lock $LOCK exists (taken by $(cat "$LOCK/owner" 2>/dev/null || echo ?)); exiting"; exit 0; fi
  # Candidate GPUs with enough free memory that no other waiter on this host claimed in the last 12 h
  # ($WORK/locks/gpu_<host>_<idx>, so two waiters that see the same card free do not both start on it).
  idx=""
  for cand in $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v m="$MIN_FREE" '$2+0 >= m {print $1}'); do
    claim="$WORK/locks/gpu_$(hostname)_$cand"
    if [ -d "$claim" ] && [ -n "$(find "$claim" -maxdepth 0 -mmin -720 2>/dev/null)" ]; then continue; fi
    idx="$cand"; break
  done
  if [ -n "${idx:-}" ]; then
    sleep 20  # a card that just freed up may be about to be taken; require it to stay free
    idx2=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v i="$idx" -v m="$MIN_FREE" '$1==i && $2+0 >= m {print $1}')
    claim="$WORK/locks/gpu_$(hostname)_$idx"
    if [ -n "${idx2:-}" ] && mkdir "$claim" 2>/dev/null; then
      if mkdir "$LOCK" 2>/dev/null; then
        echo "$(hostname) gpu=$idx $(date '+%F %T')" | tee "$LOCK/owner" > "$claim/owner"
        echo "[wait] $(date '+%F %T') got GPU $idx on $(hostname); running: $*"
        CUDA_VISIBLE_DEVICES="$idx" "$@"
        echo "[wait] $(date '+%F %T') command exited with $?"
        rm -rf "$claim"
        exit 0
      fi
      rm -rf "$claim"
    fi
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "[wait] $(date '+%F %T') gave up after $MAX_WAIT_MIN min"; exit 1; fi
  sleep 60
done

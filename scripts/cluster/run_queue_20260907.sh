#!/usr/bin/env bash
# One-card queue for the pending fp32-era experiments, in priority order (~5.5 h on a free A100-80GB):
#   1. F5   scale-up: goal+spatial x 2000 steps, OFT vs three heads, probes on goal/spatial/object, transfer to object
#   2. F3   drift-driven LLRD arm rerun with fp32 master weights (tag f3fp32)
#   3. F2   fresh PI_v3 head on the frozen F4 fp32 backbones (original protocol: batch 8, 16 diffusion repeats)
# Start through wait_for_gpu_and_run.sh so it grabs the first card with >= 62 GB free:
#   PY=<env python> bash wait_for_gpu_and_run.sh 62000 queue0912 2880 -- bash run_queue_20260907.sh
# Before every stage the card is re-checked (gpu_wait_free.sh): a job killed from outside can leave a zombie that
# still holds its memory, and a co-tenant may return -- the 2026-09-07 attempt lost every stage to exactly that cascade.
# History: 2026-09-07 lost to an external SIGKILL + zombie; 2026-09-09 F5 hung at step 374 (PyAV/dav1d thread leak in
# the DataLoader workers, fixed by starvla_lab.data.decoder_gc; run_f5_scale.sh now also kills+retries a silent run).
set -uo pipefail
WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
S="$WORK/awesome_starvla/scripts/cluster"

stage() {  # stage <name> <min_free_mib> <log_file> <command...>
  local name="$1" need="$2" log="$3"; shift 3
  if bash "$S/gpu_wait_free.sh" "$need" 90; then
    echo "[queue] $(date '+%F %T') $name: start (log $log)"
    "$@" > "$log" 2>&1
    echo "[queue] $(date '+%F %T') $name: finished (exit $?)"
  else
    echo "[queue] $(date '+%F %T') $name: SKIPPED (GPU ${CUDA_VISIBLE_DEVICES:-?} never had $need MiB free)"
  fi
}

echo "[queue] $(date '+%F %T') start on GPU ${CUDA_VISIBLE_DEVICES:-?}"
stage F5 60000 "$WORK/logs/f5_chain.log" env PY="$PY" WORK="$WORK" bash "$S/run_f5_scale.sh"
stage F3fp32 60000 "$WORK/logs/f3fp32_chain.log" env PY="$PY" WORK="$WORK" TAG=f3fp32 ARMS=driftllrd bash "$S/run_f3_llrd.sh"
stage F2PIfp32 45000 "$WORK/logs/f2fp32pi_chain.log" env PY="$PY" WORK="$WORK" bash "$S/run_f2_transfer.sh" f2 "spatial goal" "oftfp32=f4_oft_fp32 mhfp32=f4_mh_fp32" "QwenPI_v3" --trainer.lab.backbone_fp32 false
echo "[queue] ALL_DONE"

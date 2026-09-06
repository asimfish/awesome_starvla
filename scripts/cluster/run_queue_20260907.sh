#!/usr/bin/env bash
# One-card queue for the pending fp32-era experiments, in priority order (~5.5 h on a free A100-80GB):
#   1. F5   scale-up: goal+spatial x 2000 steps, OFT vs three heads, probes on goal/spatial/object, transfer to object
#   2. F3   drift-driven LLRD arm rerun with fp32 master weights (tag f3fp32)
#   3. F2   fresh PI_v3 head on the frozen F4 fp32 backbones (original protocol: batch 8, 16 diffusion repeats)
# Start through wait_for_gpu_and_run.sh so it grabs the first card with >= 62 GB free:
#   PY=<env python> bash wait_for_gpu_and_run.sh 62000 queue0907 1440 -- bash run_queue_20260907.sh
set -uo pipefail
WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
S="$WORK/awesome_starvla/scripts/cluster"

echo "[queue] $(date '+%F %T') start on GPU ${CUDA_VISIBLE_DEVICES:-?}"
PY="$PY" WORK="$WORK" bash "$S/run_f5_scale.sh" > "$WORK/logs/f5_chain.log" 2>&1; echo "[queue] $(date '+%F %T') F5 done"
PY="$PY" WORK="$WORK" TAG=f3fp32 ARMS=driftllrd bash "$S/run_f3_llrd.sh" > "$WORK/logs/f3fp32_chain.log" 2>&1; echo "[queue] $(date '+%F %T') F3 fp32 done"
PY="$PY" WORK="$WORK" bash "$S/run_f2_transfer.sh" f2 "spatial goal" "oftfp32=f4_oft_fp32 mhfp32=f4_mh_fp32" "QwenPI_v3" --trainer.lab.backbone_fp32 false > "$WORK/logs/f2fp32pi_chain.log" 2>&1; echo "[queue] $(date '+%F %T') F2 PI fp32 done"
echo "[queue] ALL_DONE"

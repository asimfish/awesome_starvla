#!/usr/bin/env bash
# F5: one notch up in scale (one GPU, ~3.5 h, needs ~60 GB): QwenOFT vs QwenMultiHead trained 2000 steps on
# LIBERO-goal + LIBERO-spatial with fp32 master weights (configs/f5_libero_2suite_2k.yaml), then
#   1. cross-head probe (pooled / retention / OFT query positions) on goal, spatial and the unseen LIBERO-object;
#   2. F2-style transfer: frozen F5 backbones (+ pretrained anchor) with a fresh OFT head, 300 steps on object.
#
#   PY=<env python> bash run_f5_scale.sh [STAGES="train probe transfer"]
set -euo pipefail

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
STAGES="${STAGES:-train probe transfer}"
S="$WORK/awesome_starvla/scripts/cluster"
F5_CONFIG="$WORK/awesome_starvla/code/starvla_lab/configs/f5_libero_2suite_2k.yaml"
MIX3="libero_goal_no_noops_1.0.0_lerobot:libero_franka,libero_spatial_no_noops_1.0.0_lerobot:libero_franka,libero_object_no_noops_1.0.0_lerobot:libero_franka"

for stage in $STAGES; do
  case "$stage" in
    train)
      echo "[f5] === f5_oft ==="
      CONFIG="$F5_CONFIG" PY="$PY" WORK="$WORK" bash "$S/run_f0_smoke.sh" QwenOFT f5_oft "$@" > "$WORK/logs/f5_oft.log" 2>&1 || echo "[f5] FAILED: f5_oft"
      echo "[f5] done: f5_oft"
      echo "[f5] === f5_mh ==="
      CONFIG="$F5_CONFIG" PY="$PY" WORK="$WORK" bash "$S/run_f0_smoke.sh" QwenMultiHead f5_mh \
        --framework.action_model.state_dim 0 --trainer.learning_rate.heads 1e-4 --trainer.learning_rate.project_layers 1e-4 \
        "$@" > "$WORK/logs/f5_mh.log" 2>&1 || echo "[f5] FAILED: f5_mh"
      echo "[f5] done: f5_mh" ;;
    probe)
      echo "[f5] === cross_head_f5 ==="
      DATA_MIX="$MIX3" N_SAMPLES=3072 EXTRA_ARGS="--query_layers 33,34,35 --num_workers 4 --pool_factor 2" PY="$PY" WORK="$WORK" \
        bash "$S/run_cross_head_probe.sh" cross_head_f5 oft5=f5_oft mh5=f5_mh > "$WORK/logs/cross_head_f5.log" 2>&1 || echo "[f5] FAILED: cross_head_f5"
      echo "[f5] done: cross_head_f5" ;;
    transfer)
      echo "[f5] === transfer to object ==="
      PY="$PY" WORK="$WORK" bash "$S/run_f2_transfer.sh" f5x "object" "pre=none oft5=f5_oft mh5=f5_mh" "QwenOFT" \
        --trainer.lab.backbone_fp32 false 2>&1 | sed 's/^/[f5] /' || echo "[f5] FAILED: transfer"
      echo "[f5] done: transfer" ;;
    *) echo "[f5] unknown stage: $stage"; exit 2 ;;
  esac
done
echo "[f5] ALL_DONE"

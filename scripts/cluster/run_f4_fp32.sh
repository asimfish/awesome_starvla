#!/usr/bin/env bash
# F4: bf16-update calibration of the single-GPU path (one GPU, 2 x ~15 min, needs ~55 / ~65 GB).
#
# Identical to the F0 v3 runs except (a) trainer.lab.backbone_fp32=true keeps the trainable backbone parameters in
# fp32 (compute stays bf16 through autocast) so AdamW updates smaller than a bf16 ulp are no longer rounded away --
# the fp32 master weights DeepSpeed keeps on the multi-GPU path -- and (b) embed_tokens is frozen (F0 v3 showed it
# costs nothing; saves ~6 GB of fp32 weight + optimizer state). Compare drift / losses with f0v3_oft_embedfrozen and
# f0v3_multihead.
#
#   PY=<env python> bash run_f4_fp32.sh [extra --dotlist overrides...]
set -euo pipefail

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
RUNNER="$WORK/awesome_starvla/scripts/cluster/run_f0_smoke.sh"

FREEZE="qwen_vl_interface.model.model.visual"
for i in $(seq 0 17); do FREEZE="$FREEZE,qwen_vl_interface.model.model.language_model.layers.$i"; done
FREEZE="$FREEZE,qwen_vl_interface.model.model.language_model.embed_tokens"
COMMON=(--trainer.freeze_modules "$FREEZE" --trainer.lab.backbone_fp32 true)

echo "[f4] === f4_oft_fp32 ==="
PY="$PY" WORK="$WORK" bash "$RUNNER" QwenOFT f4_oft_fp32 "${COMMON[@]}" "$@" > "$WORK/logs/f4_oft_fp32.log" 2>&1 || echo "[f4] FAILED: f4_oft_fp32"
echo "[f4] done: f4_oft_fp32"

echo "[f4] === f4_mh_fp32 ==="
PY="$PY" WORK="$WORK" bash "$RUNNER" QwenMultiHead f4_mh_fp32 "${COMMON[@]}" \
  --framework.action_model.state_dim 0 --trainer.learning_rate.heads 1e-4 --trainer.learning_rate.project_layers 1e-4 \
  "$@" > "$WORK/logs/f4_mh_fp32.log" 2>&1 || echo "[f4] FAILED: f4_mh_fp32"
echo "[f4] done: f4_mh_fp32"
echo "[f4] ALL_DONE"

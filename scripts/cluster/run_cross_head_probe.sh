#!/usr/bin/env bash
# WP1 cross-head probe on F0 final checkpoints (one GPU, ~15-20 min): pretrained Qwen3-VL-4B vs the fine-tuned
# backbones, action readability per suite + token-level retention. See scripts/cross_head_probe.py.
#
#   PY=<env python> bash run_cross_head_probe.sh <out_name> NAME=RUN_ID [NAME=RUN_ID ...]
#   e.g. bash run_cross_head_probe.sh cross_head_f0v3 oft=f0v3_oft multihead=f0v3_multihead oft_embedfrozen=f0v3_oft_embedfrozen
#
# Env: PY (required), WORK, DATA_ROOT, CONFIG, DATA_MIX (probe samples; default LIBERO-goal + LIBERO-spatial),
#      N_SAMPLES (default 2048), EXTRA_ARGS (appended verbatim, e.g. "--query_layers 33,34,35 --no_pooled --no_retention").
#      Uses CUDA_VISIBLE_DEVICES (default 0). Output: $WORK/results/<out_name>/.
set -euo pipefail

OUT_NAME="${1:?output name}"
shift
[ $# -ge 1 ] || { echo "need at least one NAME=RUN_ID"; exit 2; }

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
DATA_ROOT="${DATA_ROOT:-/home/dataset-assist-0/liyufeng/datasets/LEROBOT_LIBERO_DATA}"
CONFIG="${CONFIG:-$WORK/awesome_starvla/code/starvla_lab/configs/f0_libero_goal_smoke.yaml}"
DATA_MIX="${DATA_MIX:-libero_goal_no_noops_1.0.0_lerobot:libero_franka,libero_spatial_no_noops_1.0.0_lerobot:libero_franka}"
N_SAMPLES="${N_SAMPLES:-2048}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WORK/awesome_starvla/code:$WORK/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VARIANTS=()
for kv in "$@"; do
  name="${kv%%=*}"; run="${kv#*=}"
  ckpt="$WORK/checkpoints/$run/final_model/pytorch_model.pt"
  [ -f "$ckpt" ] || { echo "missing checkpoint: $ckpt"; exit 3; }
  VARIANTS+=(--variant "$name=$ckpt")
done

echo "[probe] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES out=$WORK/results/$OUT_NAME variants=$*"
cd "$WORK/starVLA"
"$PY" "$WORK/awesome_starvla/scripts/cross_head_probe.py" \
  --config "$CONFIG" --data_root_dir "$DATA_ROOT" --data_mix "$DATA_MIX" --n_samples "$N_SAMPLES" \
  --out "$WORK/results/$OUT_NAME" "${VARIANTS[@]}" ${EXTRA_ARGS:-}
echo "[probe] finished"

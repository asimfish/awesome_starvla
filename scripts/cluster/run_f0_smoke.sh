#!/usr/bin/env bash
# F0: LIBERO-goal fine-tuning smoke on one GPU through starvla_lab.train.train_starvla_lab.
#
#   PY=<env python> bash run_f0_smoke.sh <framework> [run_id] [extra --dotlist overrides...]
#   e.g.  bash run_f0_smoke.sh QwenOFT       f0_oft
#         bash run_f0_smoke.sh QwenMultiHead f0_multihead --framework.action_model.state_dim 0
#
# Env: PY (required), WORK (default /home/dataset-assist-0/liyufeng/awesome_starvla_work),
#      DATA_ROOT (default /home/dataset-assist-0/liyufeng/datasets/LEROBOT_LIBERO_DATA), CONFIG (default
#      code/starvla_lab/configs/f0_libero_goal_smoke.yaml). Uses the GPU(s) in CUDA_VISIBLE_DEVICES (default 0).
# Single process, no DeepSpeed (STARVLA_DISABLE_DEEPSPEED=1), W&B disabled; the trainer's own log goes to
# stdout, the probes to <run_dir>/lab_probes.jsonl.
set -euo pipefail

FRAMEWORK="${1:?framework name, e.g. QwenOFT or QwenMultiHead}"
RUN_ID="${2:-f0_$(echo "$FRAMEWORK" | tr '[:upper:]' '[:lower:]')}"
shift $(( $# >= 2 ? 2 : 1 ))

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
DATA_ROOT="${DATA_ROOT:-/home/dataset-assist-0/liyufeng/datasets/LEROBOT_LIBERO_DATA}"
CONFIG="${CONFIG:-$WORK/awesome_starvla/code/starvla_lab/configs/f0_libero_goal_smoke.yaml}"
RUN_ROOT="$WORK/checkpoints"
RUN_DIR="$RUN_ROOT/$RUN_ID"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WORK/awesome_starvla/code:$WORK/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export STARVLA_DISABLE_DEEPSPEED=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GR00T-style LeRobot loader needs meta/modality.json; StarVLA ships the LIBERO one under examples/.
for ds in "$DATA_ROOT"/libero_*_lerobot; do
  [ -f "$ds/meta/modality.json" ] || cp "$WORK/starVLA/examples/simBenchmarks/LIBERO/train_files/modality.json" "$ds/meta/"
done

mkdir -p "$RUN_DIR"
echo "[f0] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES framework=$FRAMEWORK run_id=$RUN_ID"
echo "[f0] starVLA @ $(git -C "$WORK/starVLA" rev-parse --short HEAD 2>/dev/null || echo unknown), awesome_starvla @ $(cat "$WORK/awesome_starvla/.git_head" 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || true  # head closes the pipe early; don't trip pipefail

cd "$WORK/starVLA"
"$PY" -m starvla_lab.train.train_starvla_lab \
  --config_yaml "$CONFIG" \
  --framework.name "$FRAMEWORK" \
  --run_id "$RUN_ID" \
  --run_root_dir "$RUN_ROOT" \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --trainer.lab.probes.jsonl_path "$RUN_DIR/lab_probes.jsonl" \
  "$@"
echo "[f0] finished; run dir: $RUN_DIR"

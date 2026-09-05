#!/usr/bin/env bash
# F2: frozen-backbone transfer (decoder lock-in at small scale), one GPU.
#
# For every (backbone, head, suite): load the backbone weights of a finished F0 run (or the pretrained VLM), freeze
# the vision encoder + all 36 LLM layers + final norm (embed_tokens stays trainable: prompt / OFT query token rows
# are treated as part of the head), attach a *fresh* action head and train it for 300 steps on one LIBERO suite.
# Within a head type the head's converged loss across backbones measures how well each backbone supports that
# head; spatial is a suite none of the backbones has seen, goal is the suite the F0 backbones were tuned on.
#
#   PY=<env python> bash run_f2_transfer.sh <tag> <suites> <backbones> <heads>
#   e.g. bash run_f2_transfer.sh f2 "spatial goal" "pre=none oft=f0v3_oft mh=f0v3_multihead" "QwenOFT QwenPI_v3"
#
# Env: PY (required), WORK, DATA_ROOT, CONFIG (default: F0 yaml), STEPS (default 300). Run ids: <tag>_<suite>_<head>_<backbone>.
set -euo pipefail

TAG="${1:?tag}"
SUITES="${2:?suites, e.g. 'spatial goal'}"
BACKBONES="${3:?backbones, e.g. 'pre=none oft=f0v3_oft mh=f0v3_multihead'}"
HEADS="${4:?heads, e.g. 'QwenOFT QwenPI_v3'}"

WORK="${WORK:-/home/dataset-assist-0/liyufeng/awesome_starvla_work}"
PY="${PY:?set PY to the StarVLA env python}"
DATA_ROOT="${DATA_ROOT:-/home/dataset-assist-0/liyufeng/datasets/LEROBOT_LIBERO_DATA}"
CONFIG="${CONFIG:-$WORK/awesome_starvla/code/starvla_lab/configs/f0_libero_goal_smoke.yaml}"
STEPS="${STEPS:-300}"

FREEZE="qwen_vl_interface.model.model.visual"
for i in $(seq 0 35); do FREEZE="$FREEZE,qwen_vl_interface.model.model.language_model.layers.$i"; done
FREEZE="$FREEZE,qwen_vl_interface.model.model.language_model.norm"

for suite in $SUITES; do
  MIX="libero_${suite}_no_noops_1.0.0_lerobot:libero_franka"
  for head in $HEADS; do
    for bb in $BACKBONES; do
      name="${bb%%=*}"; run="${bb#*=}"
      run_id="${TAG}_${suite}_$(echo "$head" | tr '[:upper:]' '[:lower:]' | tr -d '_')_${name}"
      extra=()
      if [ "$run" != "none" ]; then
        ckpt="$WORK/checkpoints/$run/final_model/pytorch_model.pt"
        [ -f "$ckpt" ] || { echo "missing checkpoint: $ckpt"; exit 3; }
        extra+=(--trainer.pretrained_checkpoint "$ckpt" --trainer.reload_modules qwen_vl_interface)
      fi
      echo "[f2] === $run_id (suite=$suite head=$head backbone=$name) ==="
      CONFIG="$CONFIG" DATA_ROOT="$DATA_ROOT" WORK="$WORK" PY="$PY" \
        bash "$WORK/awesome_starvla/scripts/cluster/run_f0_smoke.sh" "$head" "$run_id" \
          --datasets.vla_data.data_mix "$MIX" \
          --trainer.max_train_steps "$STEPS" \
          --trainer.freeze_modules "$FREEZE" \
          --trainer.learning_rate.qwen_vl_interface 1.0e-4 \
          --trainer.lab.probes.enabled false \
          "${extra[@]}" > "$WORK/logs/$run_id.log" 2>&1 || echo "[f2] FAILED: $run_id (see $WORK/logs/$run_id.log)"
      echo "[f2] done: $run_id"
    done
  done
done
echo "[f2] ALL_DONE"
